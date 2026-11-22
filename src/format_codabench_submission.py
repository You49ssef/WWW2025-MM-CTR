import csv
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "artifacts" / "submissions"
SOURCE = SUBMISSION_DIR / "baseline_transformer_residual_test_probabilities.npy"
CSV_PATH = SUBMISSION_DIR / "prediction.csv"
ZIP_PATH = SUBMISSION_DIR / "prediction.zip"
EXPECTED_ROWS = 343_025


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Prediction array not found: {SOURCE}")

    probabilities = np.load(SOURCE)
    if probabilities.ndim != 1 or probabilities.size != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} one-dimensional predictions, got "
            f"shape {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinite values")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Predicted probabilities must be in [0, 1]")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("ID", "Task1&2"))
        writer.writerows(
            (row_id, f"{float(probability):.10f}")
            for row_id, probability in enumerate(probabilities)
        )

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(CSV_PATH, arcname="prediction.csv")

    with ZIP_PATH.open("rb") as file:
        if not zipfile.is_zipfile(file):
            raise RuntimeError("The generated submission ZIP is invalid")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.namelist() != ["prediction.csv"]:
            raise RuntimeError("ZIP must contain only prediction.csv")

    print(
        f"Created {ZIP_PATH} containing prediction.csv "
        f"with {probabilities.size} rows, IDs 0..{probabilities.size - 1}, "
        f"header ID,Task1&2"
    )


if __name__ == "__main__":
    main()
