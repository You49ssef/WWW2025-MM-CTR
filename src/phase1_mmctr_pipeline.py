import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def find_dataset_dir(root: Path) -> Path:
    local = root / "data"
    if (local / "train.parquet").exists():
        return local
    candidates = (
        Path("/kaggle/input/datasets/joseph49/www2025-mm-ct"),
        Path("/kaggle/input"),
    )
    for candidate in candidates:
        if (candidate / "train.parquet").exists():
            return candidate
        match = next(candidate.glob("**/train.parquet"), None)
        if match is not None:
            return match.parent
    raise FileNotFoundError("Dataset directory was not found")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_feature_array(series, rows: int, width: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(series.to_numpy(), dtype=dtype)
    if arr.size != rows * width:
        arr = np.stack(series.to_numpy()).astype(dtype, copy=False)
    return arr.reshape(rows, width)


def build_feature_cache(data_dir: Path, out_dir: Path) -> dict:
    feature_dir = out_dir / "cache"
    feature_dir.mkdir(parents=True, exist_ok=True)

    item_info = pl.read_parquet(data_dir / "item_info.parquet")
    raw = pl.read_parquet(data_dir / "item_raw_feature.parquet")

    item_ids = item_info["item_id"].to_numpy().astype(np.int64)
    if item_ids[0] != 0 or item_ids[-1] + 1 != item_ids.size:
        raise AssertionError("item_info must contain contiguous IDs starting at zero")

    base = read_feature_array(
        item_info["item_emb_d128"], item_info.height, 128
    ).astype(np.float32)

    raw_ids = raw["item_id"].to_numpy().astype(np.int64)
    txt = read_feature_array(raw["txt_emb_BERT"], raw.height, 768)
    img = read_feature_array(raw["img_emb_CLIPRN50"], raw.height, 1024)
    if not np.array_equal(raw_ids, np.arange(1, raw.height + 1)):
        raise AssertionError("item_raw_feature IDs are not contiguous from 1")

    txt_full = np.zeros((item_info.height, 768), dtype=np.float32)
    img_full = np.zeros((item_info.height, 1024), dtype=np.float32)
    txt_full[raw_ids] = txt
    img_full[raw_ids] = img

    with torch.inference_mode():
        cpu = torch.device("cpu")
        txt_projector = nn.Linear(768, 128)
        img_projector = nn.Linear(1024, 128)
        txt_projector, img_projector = txt_projector.to(cpu), img_projector.to(cpu)
        txt_projector.load_state_dict(txt_projector.state_dict())
        img_projector.load_state_dict(img_projector.state_dict())
        txt_proj = txt_projector(torch.from_numpy(txt_full)).numpy()
        img_proj = img_projector(torch.from_numpy(img_full)).numpy()
    txt_proj = txt_proj / np.maximum(
        np.linalg.norm(txt_proj, axis=1, keepdims=True), 1e-8
    )
    img_proj = img_proj / np.maximum(
        np.linalg.norm(img_proj, axis=1, keepdims=True), 1e-8
    )
    base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)

    item_table = np.concatenate((base, txt_proj, img_proj), axis=1).astype(
        np.float32, copy=False
    )
    np.save(feature_dir / "item_table.npy", item_table)
    del txt, img, txt_full, img_full, txt_proj, img_proj, base

    tags = np.zeros((item_info.height, 5), dtype=np.int32)
    fixed_tags = np.asarray(item_info["item_tags"].to_numpy(), dtype=np.int32)
    tags[:] = fixed_tags.reshape(item_info.height, 5)
    np.save(feature_dir / "item_tags.npy", tags)

    max_user_id = 1_000_000
    max_item_id = int(item_info["item_id"].max())
    max_tag_id = int(max(tags.max(), item_info["item_tags"].explode().max()))

    for split in ("train", "valid", "test_unlabeled"):
        frame = pl.read_parquet(data_dir / f"{split}.parquet")
        cache_prefix = "test" if split == "test_unlabeled" else split

        user = frame["user_id"].to_numpy().astype(np.int32)
        item = frame["item_id"].to_numpy().astype(np.int32)
        target_ts = frame["timestamp"].to_numpy().astype(np.int64)
        likes = frame["likes_level"].to_numpy().astype(np.int16)
        views = frame["views_level"].to_numpy().astype(np.int16)

        hist_values = frame["hist_seq"].to_numpy()
        if getattr(hist_values, "dtype", None) == object:
            hist = np.stack(hist_values).astype(np.int32, copy=False)
        else:
            hist = np.asarray(hist_values, dtype=np.int32).reshape(frame.height, 50)

        hist_ts_values = frame["hist_seq_ts"].to_numpy()
        if getattr(hist_ts_values, "dtype", None) == object:
            hist_ts = np.stack(hist_ts_values).astype(np.int64, copy=False)
        else:
            hist_ts = np.asarray(hist_ts_values, dtype=np.int64).reshape(frame.height, 50)

        np.save(feature_dir / f"{cache_prefix}_user.npy", user)
        np.save(feature_dir / f"{cache_prefix}_item.npy", item)
        np.save(feature_dir / f"{cache_prefix}_timestamp.npy", target_ts)
        np.save(feature_dir / f"{cache_prefix}_hist_seq.npy", hist)
        np.save(feature_dir / f"{cache_prefix}_hist_ts.npy", hist_ts)
        np.save(feature_dir / f"{cache_prefix}_likes.npy", likes)
        np.save(feature_dir / f"{cache_prefix}_views.npy", views)

        if split != "test_unlabeled":
            label = frame["label"].to_numpy().astype(np.float32)
            np.save(feature_dir / f"{cache_prefix}_label.npy", label)

        del frame, user, item, target_ts, likes, views, hist, hist_ts
        gc.collect()

    metadata = {
        "max_user_id": max_user_id,
        "max_item_id": max_item_id,
        "max_tag_id": max_tag_id,
        "item_table_dim": 384,
        "sequence_length": 50,
    }
    (feature_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class CTRDataset(Dataset):
    def __init__(self, feature_dir: Path, split: str, include_history: bool = True):
        self.feature_dir = feature_dir
        self.split = split
        self.include_history = include_history
        self.user = np.load(feature_dir / f"{split}_user.npy", mmap_mode="r")
        self.item = np.load(feature_dir / f"{split}_item.npy", mmap_mode="r")
        self.timestamp = (
            np.load(feature_dir / f"{split}_timestamp.npy", mmap_mode="r")
            if include_history
            else None
        )
        self.hist = (
            np.load(feature_dir / f"{split}_hist_seq.npy", mmap_mode="r")
            if include_history
            else None
        )
        self.hist_ts = (
            np.load(feature_dir / f"{split}_hist_ts.npy", mmap_mode="r")
            if include_history
            else None
        )
        self.likes = np.load(feature_dir / f"{split}_likes.npy", mmap_mode="r")
        self.views = np.load(feature_dir / f"{split}_views.npy", mmap_mode="r")
        self.label = (
            None
            if split == "test"
            else np.load(feature_dir / f"{split}_label.npy", mmap_mode="r")
        )

    def __len__(self) -> int:
        return len(self.user)

    def __getitem__(self, index: int):
        result = {
            "user": torch.as_tensor(self.user[index], dtype=torch.long),
            "item": torch.as_tensor(self.item[index], dtype=torch.long),
            "likes": torch.as_tensor(self.likes[index] - 1, dtype=torch.long),
            "views": torch.as_tensor(self.views[index] - 1, dtype=torch.long),
        }
        if self.include_history:
            result["timestamp"] = torch.as_tensor(
                np.asarray(self.timestamp[index]).copy(), dtype=torch.long
            )
            result["hist"] = torch.as_tensor(
                np.asarray(self.hist[index]).copy(), dtype=torch.long
            )
            result["hist_ts"] = torch.as_tensor(
                np.asarray(self.hist_ts[index]).copy(), dtype=torch.long
            )
        if self.label is not None:
            result["label"] = torch.as_tensor(self.label[index], dtype=torch.float32)
        return result


class ItemTower(nn.Module):
    def __init__(self, item_table: np.ndarray, tag_table: np.ndarray, metadata: dict):
        super().__init__()
        item_table = torch.from_numpy(np.asarray(item_table).copy())
        tag_table = torch.from_numpy(np.asarray(tag_table).copy())
        self.register_buffer("item_table", item_table)
        self.register_buffer("tag_table", tag_table)
        self.item_projector = nn.Sequential(
            nn.Linear(metadata["item_table_dim"], 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self.tag_embedding = nn.Embedding(metadata["max_tag_id"] + 1, 16, padding_idx=0)
        self.tag_projector = nn.Linear(16, 128)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_features = self.item_table[item_ids]
        item_embedding = self.item_projector(item_features)
        tags = self.tag_table[item_ids]
        tag_embedding = self.tag_embedding(tags)
        tag_mask = tags.eq(0).unsqueeze(-1)
        tag_embedding = tag_embedding.masked_fill(tag_mask, 0)
        if tags.dim() == 2:
            tag_embedding = tag_embedding.mean(dim=1)
        else:
            tag_embedding = tag_embedding.mean(dim=2)
        return item_embedding + self.tag_projector(tag_embedding)


class TargetAwarePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(
        self, sequence: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        expanded_target = target.unsqueeze(1).expand(-1, sequence.size(1), -1)
        scores = self.attention(torch.cat((sequence, expanded_target), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


class HybridCTR(nn.Module):
    def __init__(self, architecture: str, item_table: np.ndarray, tag_table: np.ndarray,
                 metadata: dict):
        super().__init__()
        self.architecture = architecture
        dim = 128
        self.item_tower = ItemTower(item_table, tag_table, metadata)
        self.user_embedding = nn.Embedding(metadata["max_user_id"] + 1, 64)
        self.item_id_embedding = nn.Embedding(metadata["max_item_id"] + 1, 64)
        self.likes_embedding = nn.Embedding(10, 16)
        self.views_embedding = nn.Embedding(10, 16)
        self.time_embedding = nn.Embedding(96, 16)
        self.position_embedding = nn.Embedding(metadata["sequence_length"], dim)

        if architecture == "gru":
            self.sequence_encoder = nn.GRU(
                dim + 16, dim, num_layers=2, batch_first=True,
                dropout=0.10, bidirectional=False
            )
        elif architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim + 16,
                nhead=4,
                dim_feedforward=512,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
            self.sequence_projector = nn.Linear(dim + 16, dim)
        elif architecture != "baseline":
            raise ValueError(f"Unknown architecture: {architecture}")

        self.target_pool = TargetAwarePooling(dim)
        input_dim = dim * 2 + 64 + 64 + 16 + 16
        if architecture == "baseline":
            input_dim = dim + 64 + 64 + 16 + 16
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def time_buckets(self, target_ts: torch.Tensor, hist_ts: torch.Tensor) -> torch.Tensor:
        elapsed_ms = (target_ts.unsqueeze(1) - hist_ts).clamp(min=0)
        elapsed_hours = elapsed_ms.float() / 3_600_000.0
        bucket = torch.log1p(elapsed_hours).mul(12).long()
        return bucket.clamp(0, 95)

    def forward(self, batch: dict) -> torch.Tensor:
        target_item = self.item_tower(batch["item"])
        collab = torch.cat(
            (
                self.user_embedding(batch["user"]),
                self.item_id_embedding(batch["item"]),
                self.likes_embedding(batch["likes"]),
                self.views_embedding(batch["views"]),
            ),
            dim=-1,
        )

        if self.architecture == "baseline":
            features = torch.cat((target_item, collab), dim=-1)
            return self.mlp(features).squeeze(-1)

        valid_mask = batch["hist"].ne(0)
        history_item = self.item_tower(batch["hist"].clamp(min=0))
        time_tokens = self.time_embedding(
            self.time_buckets(batch["timestamp"], batch["hist_ts"])
        )
        position_ids = torch.arange(batch["hist"].size(1), device=batch["hist"].device)
        position_tokens = self.position_embedding(position_ids).unsqueeze(0)
        sequence_input = torch.cat((history_item + position_tokens, time_tokens), dim=-1)
        sequence_input = sequence_input.masked_fill(~valid_mask.unsqueeze(-1), 0)

        if self.architecture == "gru":
            lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(
                sequence_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.sequence_encoder(packed)
            sequence_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence_input.size(1),
            )
            sequence_repr = self.target_pool(
                sequence_output, target_item, valid_mask
            )
        else:
            sequence_output = self.sequence_encoder(
                sequence_input, src_key_padding_mask=~valid_mask
            )
            sequence_output = self.sequence_projector(sequence_output)
            sequence_repr = self.target_pool(
                sequence_output, target_item, valid_mask
            )

        features = torch.cat((target_item, sequence_repr, collab), dim=-1)
        if self.architecture == "gru":
            features = torch.cat((target_item, sequence_repr, collab), dim=-1)
        return self.mlp(features).squeeze(-1)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


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
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler,
                    scaler, epoch: int, best_auc: float, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_auc": best_auc,
            "config": config,
        },
        path,
    )


def train(args):
    seed_everything(args.seed)
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else find_dataset_dir(root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = out_dir / "cache"
    metadata_path = feature_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = build_feature_cache(data_dir, out_dir)
    else:
        metadata = json.loads(metadata_path.read_text())

    item_table = np.load(feature_dir / "item_table.npy", mmap_mode="r")
    tag_table = np.load(feature_dir / "item_tags.npy", mmap_mode="r")

    include_history = args.architecture != "baseline"
    train_dataset = CTRDataset(feature_dir, "train", include_history=include_history)
    valid_dataset = CTRDataset(feature_dir, "valid", include_history=include_history)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCTR(
        args.architecture, np.asarray(item_table), np.asarray(tag_table), metadata
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / f"{args.architecture}_history.json"
    history = []
    best_auc = -1.0

    start_epoch = 0
    latest_path = checkpoint_dir / f"{args.architecture}_latest.pt"
    if latest_path.exists() and args.resume:
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint["best_auc"])
        if history_path.exists():
            history = json.loads(history_path.read_text())
        print(f"Resumed {args.architecture} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        count = 0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda"
            ):
                logits = model(batch)
                loss = loss_fn(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_count = batch["label"].size(0)
            running_loss += float(loss.detach()) * batch_count
            count += batch_count
            if step and step % args.log_every == 0:
                print(
                    f"{args.architecture} epoch={epoch + 1} step={step} "
                    f"loss={running_loss / count:.6f}"
                )

        metrics = evaluate(model, valid_loader, device)
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(count, 1),
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "accuracy": metrics["accuracy"],
            "seconds": elapsed,
            "rows": count,
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))
        print(json.dumps(record, indent=2))

        config = vars(args).copy()
        config["data_dir"] = str(data_dir)
        save_checkpoint(
            latest_path, model, optimizer, scheduler, scaler,
            epoch, best_auc, config
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            save_checkpoint(
                checkpoint_dir / f"{args.architecture}_best.pt",
                model, optimizer, scheduler, scaler, epoch, best_auc, config
            )
            print(f"Saved new best {args.architecture} checkpoint: AUC={best_auc:.6f}")

    write_predictions(
        args, model, feature_dir, device, out_dir,
        checkpoint_dir / f"{args.architecture}_best.pt"
    )


@torch.inference_mode()
def write_predictions(args, model: nn.Module, feature_dir: Path,
                      device: torch.device, out_dir: Path, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dataset = CTRDataset(
        feature_dir,
        "test",
        include_history=args.architecture != "baseline",
    )
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)
    probabilities = []
    for batch in test_loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits = model(batch)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    probabilities = np.concatenate(probabilities)
    probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)

    submission_dir = out_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{args.architecture}_submission.csv"
    np.savetxt(
        submission_path,
        probabilities,
        delimiter="\n",
        fmt="%.10f",
    )

    np.save(submission_dir / f"{args.architecture}_test_probabilities.npy", probabilities)
    print(
        f"Saved test predictions: {submission_path} "
        f"rows={probabilities.size} range=({probabilities.min():.6f}, "
        f"{probabilities.max():.6f})"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--out_dir", default=str(Path(__file__).resolve().parent / "artifacts")
    )
    parser.add_argument(
        "--architecture",
        choices=("baseline", "gru", "transformer"),
        default="gru",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
