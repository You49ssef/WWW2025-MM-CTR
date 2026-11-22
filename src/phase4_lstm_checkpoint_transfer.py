import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from phase1_mmctr_pipeline import CTRDataset, HybridCTR, make_loader
from phase3_checkpoint_transfer import BaselineResidualSequential


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
CACHE = OUT / "cache"
GRU_CHECKPOINT = OUT / "checkpoints" / "baseline_gru_residual_best.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMResidualModel(nn.Module):
    def __init__(
        self,
        frozen_gru_residual: BaselineResidualSequential,
        sequence_length: int,
    ):
        super().__init__()
        self.frozen_gru_residual = frozen_gru_residual
        for parameter in self.frozen_gru_residual.parameters():
            parameter.requires_grad_(False)
        self.frozen_gru_residual.eval()

        dim = 128
        self.time_embedding = nn.Embedding(96, 16)
        self.position_embedding = nn.Embedding(sequence_length, dim)
        self.lstm = nn.LSTM(
            dim + 16,
            dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.target_pool = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1),
            requires_grad=True,
        )

    def time_buckets(
        self,
        target_ts: torch.Tensor,
        history_ts: torch.Tensor,
    ) -> torch.Tensor:
        elapsed_ms = (target_ts.unsqueeze(1) - history_ts).clamp(min=0)
        elapsed_hours = elapsed_ms.float() / 3_600_000.0
        return torch.log1p(elapsed_hours).mul(12).long().clamp(0, 95)

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            prior_logit = self.frozen_gru_residual(batch)

        valid_mask = batch["hist"].ne(0)
        history_item = self.frozen_gru_residual.baseline.item_tower(
            batch["hist"].clamp(min=0)
        )
        time_tokens = self.time_embedding(
            self.time_buckets(batch["timestamp"], batch["hist_ts"])
        )
        positions = torch.arange(
            batch["hist"].size(1),
            device=batch["hist"].device,
        )
        position_tokens = self.position_embedding(positions).unsqueeze(0)
        sequence_input = torch.cat(
            (
                history_item + position_tokens,
                time_tokens,
            ),
            dim=-1,
        )
        sequence_input = sequence_input.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0,
        )

        lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            sequence_input,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        sequence_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=batch["hist"].size(1),
        )

        target_item = self.frozen_gru_residual.baseline.item_tower(batch["item"])
        target_expanded = target_item.unsqueeze(1).expand(
            -1,
            sequence_output.size(1),
            -1,
        )
        attention_input = torch.cat(
            (sequence_output, target_expanded),
            dim=-1,
        )
        scores = self.target_pool(attention_input).squeeze(-1)
        scores = scores.masked_fill(
            ~valid_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=1)
        sequence_repr = torch.bmm(
            weights.unsqueeze(1),
            sequence_output,
        ).squeeze(1)

        residual = self.residual_head(sequence_repr).squeeze(-1)
        return prior_logit + self.residual_scale * residual


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits = []
    labels = []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        logits.append(model(batch).float().cpu().numpy())
        labels.append(batch["label"].float().cpu().numpy())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


@torch.inference_mode()
def write_test_predictions(model, device, batch_size, amp):
    test_dataset = CTRDataset(CACHE, "test", include_history=True)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False)
    model.eval()
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
        submission_dir / "baseline_gru_lstm_residual_test_probabilities.npy",
        probabilities,
    )
    np.savetxt(
        submission_dir / "baseline_gru_lstm_residual_submission.csv",
        probabilities,
        fmt="%.10f",
    )
    print(
        "Saved LSTM residual test predictions:",
        len(probabilities),
        float(probabilities.min()),
        float(probabilities.max()),
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not GRU_CHECKPOINT.exists():
        raise FileNotFoundError(GRU_CHECKPOINT)

    metadata = json.loads((CACHE / "metadata.json").read_text())
    item_table = np.load(CACHE / "item_table.npy", mmap_mode="r")
    tag_table = np.load(CACHE / "item_tags.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = HybridCTR(
        "baseline",
        np.asarray(item_table),
        np.asarray(tag_table),
        metadata,
    ).to(device)
    gru_residual = BaselineResidualSequential(
        baseline,
        "gru",
        metadata["sequence_length"],
    ).to(device)
    checkpoint = torch.load(GRU_CHECKPOINT, map_location=device)
    gru_residual.load_state_dict(checkpoint["model_state"], strict=True)
    gru_residual.eval()

    model = LSTMResidualModel(
        gru_residual,
        metadata["sequence_length"],
    ).to(device)

    checkpoint_path = OUT / "checkpoints" / "baseline_gru_lstm_residual_best.pt"
    best_lstm_checkpoint = (
        torch.load(checkpoint_path, map_location=device)
        if checkpoint_path.exists()
        else None
    )
    if args.predict:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model.load_state_dict(best_lstm_checkpoint["model_state"])
        model.eval()
        write_test_predictions(model, device, args.batch_size, args.amp)
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
    history = []
    best_auc = float(
        best_lstm_checkpoint["best_auc"]
        if best_lstm_checkpoint is not None
        else checkpoint["best_auc"]
    )
    start_epoch = 0
    history_path = OUT / "baseline_gru_lstm_residual_history.json"
    latest_path = OUT / "checkpoints" / "baseline_gru_lstm_residual_latest.pt"

    if args.resume:
        if history_path.exists():
            history = json.loads(history_path.read_text())
        if latest_path.exists():
            saved = torch.load(latest_path, map_location=device)
            model.load_state_dict(saved["model_state"], strict=True)
            if saved.get("optimizer_state") is not None:
                optimizer.load_state_dict(saved["optimizer_state"])
            if saved.get("scaler_state") is not None:
                scaler.load_state_dict(saved["scaler_state"])
            start_epoch = int(saved.get("epoch", -1)) + 1
        else:
            if best_lstm_checkpoint is None:
                raise FileNotFoundError(checkpoint_path)
            model.load_state_dict(
                best_lstm_checkpoint["model_state"],
                strict=True,
            )
            start_epoch = max(
                (int(record["epoch"]) for record in history),
                default=int(best_lstm_checkpoint.get("epoch", -1)) + 1,
            )
        print(
            f"Resumed LSTM residual at epoch {start_epoch + 1}; "
            f"best AUC={best_auc:.6f}"
        )

    for epoch in range(start_epoch, start_epoch + args.epochs):
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
                    f"LSTM step={step} "
                    f"loss={running_loss / rows:.6f}"
                )

        metrics = evaluate(model, valid_loader, device)
        record = {
            "epoch": epoch + 1,
            "architecture": "baseline_gru_lstm_residual",
            "train_loss": running_loss / max(rows, 1),
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "rows": rows,
            "seconds": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        history_path.write_text(json.dumps(history, indent=2))

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_auc": best_auc,
                "architecture": record["architecture"],
                "config": vars(args),
            },
            latest_path,
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "gru_checkpoint": str(GRU_CHECKPOINT),
                    "architecture": record["architecture"],
                    "best_auc": best_auc,
                    "config": vars(args),
                },
                checkpoint_path,
            )

    history_path.write_text(json.dumps(history, indent=2))
    if checkpoint_path.exists():
        best = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best["model_state"])
        model.eval()
    write_test_predictions(model, device, args.batch_size, args.amp)


if __name__ == "__main__":
    main()
