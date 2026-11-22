import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from phase1_mmctr_pipeline import CTRDataset, HybridCTR, make_loader


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
CACHE = OUT / "cache"
CHECKPOINT = OUT / "checkpoints" / "baseline_best.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineResidualSequential(nn.Module):
    def __init__(
        self,
        baseline: HybridCTR,
        architecture: str,
        sequence_length: int,
        dim: int = 128,
    ):
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)

        self.time_embedding = nn.Embedding(96, 16)
        self.position_embedding = nn.Embedding(sequence_length, dim)
        if architecture == "gru":
            self.sequence_encoder = nn.GRU(
                dim + 16,
                dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.sequence_pool = nn.Linear(dim, dim)
        elif architecture == "transformer":
            encoder = nn.TransformerEncoderLayer(
                d_model=dim + 16,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder,
                num_layers=1,
            )
            self.sequence_projector = nn.Linear(dim + 16, dim)
            self.sequence_pool = nn.Linear(dim, dim)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.residual_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.architecture = architecture
        self.register_buffer(
            "residual_scale",
            torch.tensor(0.1),
            persistent=True,
        )

    def time_buckets(
        self,
        target_ts: torch.Tensor,
        history_ts: torch.Tensor,
    ) -> torch.Tensor:
        elapsed_ms = (target_ts.unsqueeze(1) - history_ts).clamp(min=0)
        elapsed_hours = elapsed_ms.float() / 3_600_000.0
        bucket = torch.log1p(elapsed_hours).mul(12).long()
        return bucket.clamp(0, 95)

    def forward(self, batch: dict) -> torch.Tensor:
        baseline_logit = self.baseline(batch)

        valid_mask = batch["hist"].ne(0)
        history_item = self.baseline.item_tower(batch["hist"].clamp(min=0))
        time_token = self.time_embedding(
            self.time_buckets(batch["timestamp"], batch["hist_ts"])
        )
        position = torch.arange(
            batch["hist"].size(1),
            device=batch["hist"].device,
        )
        position_token = self.position_embedding(position).unsqueeze(0)
        sequence_input = torch.cat(
            (
                history_item + position_token,
                time_token,
            ),
            dim=-1,
        )
        sequence_input = sequence_input.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0,
        )

        if self.architecture == "gru":
            lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence_input,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence_encoder(packed)
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=batch["hist"].size(1),
            )
        else:
            sequence_output = self.sequence_encoder(
                sequence_input,
                src_key_padding_mask=~valid_mask,
            )
            sequence_output = self.sequence_projector(sequence_output)

        target_item = self.baseline.item_tower(batch["item"])
        target_expanded = target_item.unsqueeze(1).expand(
            -1,
            sequence_output.size(1),
            -1,
        )
        attention = torch.einsum(
            "btd,btd->bt",
            sequence_output,
            target_expanded,
        )
        attention = attention.masked_fill(
            ~valid_mask,
            torch.finfo(attention.dtype).min,
        )
        weights = torch.softmax(attention, dim=1)
        sequence_repr = torch.bmm(
            weights.unsqueeze(1),
            sequence_output,
        ).squeeze(1)
        sequence_repr = self.sequence_pool(sequence_repr)

        residual = self.residual_head(sequence_repr).squeeze(-1)
        return baseline_logit.detach() + self.residual_scale * residual


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits = []
    labels = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits.append(model(batch).float().cpu().numpy())
        labels.append(batch["label"].float().cpu().numpy())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint is missing: {CHECKPOINT}"
        )

    metadata = json.loads((CACHE / "metadata.json").read_text())
    item_table = np.load(CACHE / "item_table.npy", mmap_mode="r")
    tag_table = np.load(CACHE / "item_tags.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = HybridCTR(
        "baseline",
        np.asarray(item_table),
        np.asarray(tag_table),
        metadata,
    )
    baseline_checkpoint = torch.load(CHECKPOINT, map_location=device)
    baseline.load_state_dict(baseline_checkpoint["model_state"])
    baseline.to(device)
    baseline.eval()

    model = BaselineResidualSequential(
        baseline,
        args.architecture,
        metadata["sequence_length"],
    ).to(device)

    if args.predict_only:
        residual_checkpoint = (
            OUT / "checkpoints"
            / f"baseline_{args.architecture}_residual_best.pt"
        )
        if not residual_checkpoint.exists():
            raise FileNotFoundError(
                f"Residual checkpoint is missing: {residual_checkpoint}"
            )
        write_test_predictions(
            residual_checkpoint,
            args.architecture,
            baseline,
            metadata,
            device,
            args.batch_size,
            args.amp,
        )
        return

    train_dataset = CTRDataset(CACHE, "train", include_history=True)
    valid_dataset = CTRDataset(CACHE, "valid", include_history=True)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_dataset, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        rows = 0
        start = time.time()
        for step, batch in enumerate(train_loader):
            if step >= args.max_steps:
                break
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(batch)
                loss = loss_fn(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_rows = batch["label"].size(0)
            running_loss += float(loss.detach()) * batch_rows
            rows += batch_rows
            if step and step % 200 == 0:
                print(
                    f"{args.architecture} step={step} "
                    f"loss={running_loss / rows:.6f}"
                )

        metrics = evaluate(model, valid_loader, device)
        record = {
            "epoch": epoch + 1,
            "architecture": f"baseline_{args.architecture}_residual",
            "train_loss": running_loss / max(rows, 1),
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "rows": rows,
            "seconds": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "baseline_checkpoint": str(CHECKPOINT),
                    "architecture": record["architecture"],
                    "best_auc": best_auc,
                    "config": vars(args),
                },
                checkpoint_dir / f"baseline_{args.architecture}_residual_best.pt",
            )

    (OUT / f"baseline_{args.architecture}_residual_history.json").write_text(
        json.dumps(history, indent=2)
    )

    residual_checkpoint = (
        checkpoint_dir / f"baseline_{args.architecture}_residual_best.pt"
    )
    write_test_predictions(
        residual_checkpoint,
        args.architecture,
        baseline,
        metadata,
        device,
        args.batch_size,
        args.amp,
    )


@torch.inference_mode()
def write_test_predictions(
    checkpoint_path: Path,
    architecture: str,
    baseline: HybridCTR,
    metadata: dict,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> None:
    model = BaselineResidualSequential(
        baseline,
        architecture,
        metadata["sequence_length"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dataset = CTRDataset(CACHE, "test", include_history=True)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False)
    probabilities = []
    for batch in test_loader:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.amp.autocast(
            "cuda",
            enabled=amp and device.type == "cuda",
        ):
            logits = model(batch)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())

    probabilities = np.concatenate(probabilities)
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    submission_dir = OUT / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        submission_dir / f"baseline_{architecture}_residual_test_probabilities.npy",
        probabilities,
    )
    np.savetxt(
        submission_dir / f"baseline_{architecture}_residual_submission.csv",
        probabilities,
        fmt="%.10f",
    )
    print(
        f"Saved residual test predictions: rows={len(probabilities)} "
        f"range=({probabilities.min():.6f}, {probabilities.max():.6f})"
    )


if __name__ == "__main__":
    main()
