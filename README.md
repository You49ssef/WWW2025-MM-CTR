# WWW2025 MM-CTR

Code and selected results for the WWW2025 multimodal CTR prediction project.

## Contents

- `src/`: data audit, training, checkpoint transfer, and submission-format scripts.
- `results/`: validation metrics and training histories.
- `submission/`: Codabench-ready `prediction.csv` and `prediction.zip`.

The dataset and model checkpoints are intentionally not included. Dataset files
are large, and the checkpoints exceed GitHub's 100 MB per-file limit. Keep the
dataset locally or in Kaggle. If checkpoints must be versioned, use Git LFS or
an appropriate model-artifact store.

## Data

Place the five Parquet files in `src/data/`:

- `train.parquet`
- `valid.parquet`
- `test_unlabeled.parquet`
- `item_info.parquet`
- `item_raw_feature.parquet`

Do not use the test labels; they are unavailable. The provided preprocessing
preserves item ID 0 as padding and keeps test predictions in input row order.

## Environment

Use Python 3.10+ and install dependencies:

```bash
python -m pip install -r requirements.txt
```

The training scripts use CUDA when available and otherwise fall back to CPU.
For large-scale training, a CUDA GPU is strongly recommended.

## Workflow

Run the dataset audit:

```bash
python src/phase0_dataset_audit.py
```

Train the baseline:

```bash
python src/phase1_mmctr_pipeline.py --data_dir src/data --out_dir src/artifacts --architecture baseline --epochs 1 --batch_size 512
```

Train a sequential residual model using the baseline checkpoint:

```bash
python src/phase3_checkpoint_transfer.py --architecture transformer --epochs 5 --batch_size 128 --lr 0.0001 --max_steps 1000
```

Train the LSTM residual model using the GRU residual checkpoint:

```bash
python src/phase4_lstm_checkpoint_transfer.py --epochs 5 --batch_size 128 --lr 0.0001 --max_steps 1000
```

Training artifacts are written under `src/artifacts/`.
The checkpoint-transfer scripts require their preceding checkpoints to exist.

## Validation results

Validation results from the recorded experiments:

| Model | Training budget | AUC | LogLoss |
|---|---:|---:|---:|
| Baseline | Full train set, 1 epoch | 0.644562 | 0.479812 |
| Baseline + Transformer residual | 5 × 1,000 batches | 0.648354 | 0.478617 |
| Baseline + GRU + LSTM residual | 30 additional × 1,000 batches | 0.645339 | 0.479560 |

These are local validation results, not competition leaderboard scores. Each
1,000-batch run uses 128 examples per batch; it is not a full pass over the
6.15-million-row training set.

## Codabench submission

`submission/prediction.zip` contains `prediction.csv` with columns
`ID,Task1&2` and 343,025 rows. The IDs are sequential (`0` through `343024`)
in test input order. The submission was generated from the best validation
Transformer residual checkpoint.

To reformat predictions after generating
`src/artifacts/submissions/baseline_transformer_residual_test_probabilities.npy`,
run:

```bash
python src/format_codabench_submission.py
```
