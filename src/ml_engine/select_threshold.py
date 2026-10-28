
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_engine.autoencoder import PacketAutoencoder


# Project paths
ROOT = PROJECT_ROOT

DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
CURRENT_THRESHOLD_PATH = ROOT / "models" / "threshold.json"
OUTPUT_PATH = ROOT / "reports" / "candidate_threshold.json"

# Keep the same 10 features used by the trained model.
FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    "Flow Bytes/s",
    " Flow Packets/s",
    " SYN Flag Count",
    " ACK Flag Count",
    " Average Packet Size",
]

TARGET_FPR = 0.05
RANDOM_STATE = 42
BATCH_SIZE = 2048


def _load_current_threshold(path: Path):
    if not path.exists():
        return {"threshold": None, "note": "No current threshold file found."}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("AegisAgent - Threshold Selection")
    print("-" * 45)

    # Check required files before starting.
    for path in [
        DATA_PATH,
        MODEL_PATH,
        SCALER_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load the dataset.
    print("Loading CICIDS2017 dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    # Find the label column.
    label_column = "Label"
    if label_column not in df.columns:
        raise ValueError(
            f"Could not find '{label_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Check that all model features exist.
    missing = [feature.strip() for feature in FEATURES
               if feature.strip() not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Keep benign traffic only.
    benign = df[
        df[label_column].astype(str).str.strip().str.upper() == "BENIGN"
    ].copy()

    if len(benign) < 100:
        raise ValueError(
            f"Not enough benign rows found: {len(benign)}"
        )

    # Reproduce the training split exactly.
    _, benign_validation = train_test_split(
        benign,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Clean and validate feature values.
    X = benign_validation[
        [feature.strip() for feature in FEATURES]
    ].apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    if X.empty:
        raise ValueError("No valid benign validation rows remain.")

    print(f"Benign validation rows: {len(X):,}")

    # Load the scaler used during training.
    import joblib
    scaler = joblib.load(SCALER_PATH)

    X_scaled = scaler.transform(X)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the trained autoencoder.
    model = PacketAutoencoder(input_dim=len(FEATURES)).to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
        )
    model.load_state_dict(state_dict)
    model.eval()

    # Calculate reconstruction errors in batches.
    errors = []

    with torch.no_grad():
        for start in range(0, len(X_scaled), BATCH_SIZE):
            batch = torch.tensor(
                X_scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )

            reconstructed = model(batch)
            batch_errors = torch.mean(
                (reconstructed - batch) ** 2,
                dim=1,
            )

            errors.extend(batch_errors.cpu().numpy().tolist())

    errors = np.asarray(errors, dtype=np.float64)

    if not np.isfinite(errors).all():
        raise ValueError("Non-finite reconstruction errors found.")

    # Select a threshold based only on benign traffic.
    # The 95th percentile targets approximately 5% FPR.
    candidate_threshold = float(np.quantile(errors, 1 - TARGET_FPR))
    actual_fpr = float(np.mean(errors > candidate_threshold))

    # Read the existing threshold for comparison when present.
    current_data = _load_current_threshold(CURRENT_THRESHOLD_PATH)

    # Save a separate candidate; do not modify threshold.json.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "candidate_threshold": candidate_threshold,
        "target_false_positive_rate": TARGET_FPR,
        "measured_benign_false_positive_rate": actual_fpr,
        "benign_validation_rows": int(len(errors)),
        "score_median": float(np.median(errors)),
        "score_p95": float(np.quantile(errors, 0.95)),
        "score_p99": float(np.quantile(errors, 0.99)),
        "current_threshold_file": str(CURRENT_THRESHOLD_PATH),
        "current_threshold_data": current_data,
        "note": (
            "Candidate only. Existing models/threshold.json was not changed. "
            "This validation split may already have been used during "
            "training threshold calibration; use an independent benign "
            "dataset for a stronger final assessment."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nThreshold selection completed.")
    print(f"Target FPR:              {TARGET_FPR:.1%}")
    print(f"Measured validation FPR:  {actual_fpr:.2%}")
    print(f"Candidate threshold:      {candidate_threshold:.10g}")
    print(f"Current threshold data:   {current_data}")
    print(f"Candidate saved to:       {OUTPUT_PATH}")
    print("\nYour existing threshold.json was not changed.")


if __name__ == "__main__":
    main()
