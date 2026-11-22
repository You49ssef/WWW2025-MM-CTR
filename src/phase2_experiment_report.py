import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from phase1_mmctr_pipeline import (
    CTRDataset,
    HybridCTR,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
CACHE = OUT / "cache"
CHECKPOINTS = OUT / "checkpoints"
SUBMISSIONS = OUT / "submissions"


def load_model(architecture: str, device: torch.device) -> torch.nn.Module:
    metadata = json.loads((CACHE / "metadata.json").read_text())
    item_table = np.load(CACHE / "item_table.npy", mmap_mode="r")
    tag_table = np.load(CACHE / "item_tags.npy", mmap_mode="r")
    model = HybridCTR(
        architecture,
        np.asarray(item_table),
        np.asarray(tag_table),
        metadata,
    ).to(device)
    checkpoint = torch.load(
        CHECKPOINTS / f"{architecture}_best.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.inference_mode()
def validation_probabilities(
    architecture: str, device: torch.device, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    model = load_model(architecture, device)
    dataset = CTRDataset(
        CACHE,
        "valid",
        include_history=architecture != "baseline",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logits = []
    labels = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits.append(model(batch).float().cpu().numpy())
        labels.append(batch["label"].float().cpu().numpy())
    probabilities = torch.sigmoid(
        torch.from_numpy(np.concatenate(logits))
    ).numpy()
    labels = np.concatenate(labels)
    del model
    torch.cuda.empty_cache()
    return probabilities, labels


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def write_experiment_table() -> None:
    experiments = [
        {
            "experiment": "E1_baseline_smoke",
            "architecture": "baseline",
            "steps": 200,
            "auc": 0.531474160697105,
            "logloss": 0.499415725469589,
            "accuracy": 0.8,
        },
        {
            "experiment": "E1_baseline_full",
            "architecture": "baseline",
            "steps": 12000,
            "auc": 0.6445615988637855,
            "logloss": 0.47981178760528564,
            "accuracy": 0.7999235788607372,
        },
        {
            "experiment": "E2_gru_smoke",
            "architecture": "gru",
            "steps": 100,
            "auc": 0.5187894590723439,
            "logloss": 0.5023441910743713,
            "accuracy": 0.8,
        },
        {
            "experiment": "E2_gru_partial",
            "architecture": "gru",
            "steps": 1000,
            "auc": 0.5398430915179783,
            "logloss": 0.5017349720001221,
            "accuracy": 0.8,
        },
        {
            "experiment": "E3_transformer_smoke",
            "architecture": "transformer",
            "steps": 100,
            "auc": 0.5151643044591313,
            "logloss": 0.5019369721412659,
            "accuracy": 0.8,
        },
        {
            "experiment": "E3_transformer_partial",
            "architecture": "transformer",
            "steps": 500,
            "auc": 0.5348831332821569,
            "logloss": 0.4993314146995544,
            "accuracy": 0.8,
        },
    ]
    experiments_path = OUT / "experiments.json"
    experiments_path.write_text(json.dumps(experiments, indent=2))

    with (OUT / "experiments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("experiment", "architecture", "steps", "auc", "logloss", "accuracy"),
        )
        writer.writeheader()
        writer.writerows(experiments)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_experiment_table()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_predictions = {}
    labels = None
    for architecture in ("baseline", "gru", "transformer"):
        probabilities, current_labels = validation_probabilities(architecture, device)
        valid_predictions[architecture] = probabilities
        if labels is None:
            labels = current_labels
        np.save(OUT / f"{architecture}_valid_probabilities.npy", probabilities)
        print(architecture, json.dumps(metrics(labels, probabilities)))

    ensemble_results = []
    for architecture, probabilities in valid_predictions.items():
        if architecture == "baseline":
            continue
        for weight in np.arange(0.1, 1.0, 0.1):
            ensemble = weight * valid_predictions["baseline"] + (
                1 - weight
            ) * probabilities
            ensemble_results.append(
                {
                    "model_a": "baseline",
                    "model_b": architecture,
                    "weight_baseline": float(weight),
                    "weight_model_b": float(1 - weight),
                    **metrics(labels, ensemble),
                }
            )

    (OUT / "ensemble_validation.json").write_text(
        json.dumps(ensemble_results, indent=2)
    )
    best = max(ensemble_results, key=lambda result: result["auc"])
    print("best_ensemble", json.dumps(best, indent=2))
    baseline_metrics = metrics(
        labels,
        valid_predictions["baseline"],
    )
    if best["auc"] > baseline_metrics["auc"] and best["logloss"] < baseline_metrics["logloss"]:
        baseline_test = np.load(SUBMISSIONS / "baseline_test_probabilities.npy")
        model_test = np.load(
            SUBMISSIONS / f"{best['model_b']}_test_probabilities.npy"
        )
        ensemble_test = (
            best["weight_baseline"] * baseline_test
            + best["weight_model_b"] * model_test
        )
        np.save(SUBMISSIONS / "ensemble_test_probabilities.npy", ensemble_test)
        np.savetxt(
            SUBMISSIONS / "ensemble_submission.csv",
            ensemble_test,
            fmt="%.10f",
        )
        print(
            "validated_ensemble_test",
            len(ensemble_test),
            float(ensemble_test.min()),
            float(ensemble_test.max()),
        )
    else:
        print("No ensemble passed strict validation improvement; baseline is the submission candidate.")


if __name__ == "__main__":
    main()
