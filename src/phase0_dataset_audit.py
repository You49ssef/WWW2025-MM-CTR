from pathlib import Path
import platform
import sys

import polars as pl
import numpy as np


KAGGLE_DATASETS = [
    Path("/kaggle/input/datasets/joseph49/www2025-mm-ct"),
    Path("/kaggle/input"),
]
LOCAL_DATA = Path(__file__).resolve().parent / "data"


def find_dataset_dir() -> Path:
    for candidate in KAGGLE_DATASETS:
        if candidate.exists() and (candidate / "train.parquet").exists():
            return candidate
        kaggle_match = next(candidate.glob("**/train.parquet"), None)
        if kaggle_match is not None:
            return kaggle_match.parent
    if (LOCAL_DATA / "train.parquet").exists():
        return LOCAL_DATA
    raise FileNotFoundError(
        "Could not find train.parquet. Set DATA_DIR explicitly to the Kaggle "
        "dataset directory or the extracted local data directory."
    )


DATA_DIR = find_dataset_dir()
REQUIRED_FILES = (
    "train.parquet",
    "valid.parquet",
    "test_unlabeled.parquet",
    "item_info.parquet",
    "item_raw_feature.parquet",
)
missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Dataset directory: {DATA_DIR}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as exc:
    print(f"PyTorch inspection failed: {type(exc).__name__}: {exc}")

train = pl.read_parquet(DATA_DIR / "train.parquet")
valid = pl.read_parquet(DATA_DIR / "valid.parquet")
test = pl.read_parquet(DATA_DIR / "test_unlabeled.parquet")
item_info = pl.read_parquet(DATA_DIR / "item_info.parquet")
item_raw = pl.read_parquet(DATA_DIR / "item_raw_feature.parquet")

print("\nSchemas:")
for name, frame in (
    ("train", train),
    ("valid", valid),
    ("test", test),
    ("item_info", item_info),
    ("item_raw_feature", item_raw),
):
    print(name, frame.shape, frame.schema)

required_train_columns = {
    "user_id",
    "item_id",
    "timestamp",
    "hist_seq",
    "hist_seq_ts",
    "label",
    "likes_level",
    "views_level",
}
for name, frame in (("train", train), ("valid", valid), ("test", test)):
    missing_columns = required_train_columns - set(frame.columns)
    if missing_columns:
        raise AssertionError(f"{name} is missing columns: {sorted(missing_columns)}")

assert train["item_id"].min() >= 1
assert item_info["item_id"].n_unique() == item_info.height
assert item_info.row(0, named=True)["item_id"] == 0
assert item_info.row(0, named=True)["item_emb_d128"] == [0.0] * 128
assert item_raw["item_id"].n_unique() == item_raw.height

print("\nCore checks:")
for name, frame in (("train", train), ("valid", valid), ("test", test)):
    hist_lens = frame["hist_seq"].list.len()
    ts_lens = frame["hist_seq_ts"].list.len()
    mismatch = int((hist_lens != ts_lens).sum())
    unknown_items = frame.join(
        item_info.select("item_id"), on="item_id", how="anti"
    ).height
    print(
        name,
        {
            "rows": frame.height,
            "hist_length_range": (int(hist_lens.min()), int(hist_lens.max())),
            "history_timestamp_length_mismatch": mismatch,
            "unknown_candidate_items": unknown_items,
            "null_count": sum(frame.null_count().row(0)),
        },
    )

print("\nLabel distribution:")
print(train["label"].value_counts().sort("label"))
print(valid["label"].value_counts().sort("label"))
print("Test labels:", int(test["label"].null_count()), "/", test.height)

raw_ids = set(item_raw["item_id"].to_list())
info_ids = set(item_info["item_id"].to_list()) - {0}
assert raw_ids == info_ids
print("\nItem feature alignment:")
print(
    {
        "item_info_rows": item_info.height,
        "item_raw_rows": item_raw.height,
        "padding_row": item_info.row(0, named=True)["item_id"],
        "raw_info_sets_equal": raw_ids == info_ids,
    }
)

train_items = set(train["item_id"].to_list())
valid_items = set(valid["item_id"].to_list())
test_items = set(test["item_id"].to_list())
print("\nCandidate-item coverage:")
print(
    {
        "valid_items_unseen_in_train": len(valid_items - train_items),
        "test_items_unseen_in_train": len(test_items - train_items),
        "train_users": train["user_id"].n_unique(),
        "valid_users": valid["user_id"].n_unique(),
        "test_users": test["user_id"].n_unique(),
    }
)

sample_size = min(100_000, train.height)
sample = train.select("timestamp", "hist_seq", "hist_seq_ts").head(sample_size)
future_count = 0
decreasing_count = 0
zero_count = []
for target_ts, history, history_ts in sample.iter_rows():
    history = [int(value) for value in history]
    history_ts = [int(value) for value in history_ts]
    zero_count.append(history.count(0))
    future_count += int(any(value > int(target_ts) for value in history_ts))
    decreasing_count += int(
        any(right < left for left, right in zip(history_ts, history_ts[1:]))
    )

print("\nTemporal and padding sample:")
print(
    {
        "sample_rows": sample_size,
        "history_padding_zero_range": (min(zero_count), max(zero_count)),
        "mean_history_padding": float(np.mean(zero_count)),
        "history_timestamp_after_target": future_count,
        "history_timestamp_not_increasing": decreasing_count,
    }
)
