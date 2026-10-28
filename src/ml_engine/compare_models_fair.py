
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.ml_engine.autoencoder import PacketAutoencoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
REPORT_PATH = ROOT / "reports" / "fair_model_comparison.json"

FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
TARGET_FPR = 0.05
RANDOM_STATE = 2026


def load_model(model_path):
    model = PacketAutoencoder(input_dim=len(FEATURES)).to(DEVICE)

    try:
        state = torch.load(
            model_path, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()
    return model


def calculate_scores(model, scaler, frame):
    """Calculate one reconstruction-error score per flow."""
    values = frame[FEATURES].astype(np.float64)
    scaled = scaler.transform(values)

    scores = []

    with torch.no_grad():
        for start in range(0, len(scaled), BATCH_SIZE):
            batch = torch.as_tensor(
                scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )

            reconstructed = model(batch)
            errors = torch.mean(
                (reconstructed - batch) ** 2, dim=1
            )
            scores.extend(errors.cpu().numpy().tolist())

    return np.asarray(scores)


def evaluate(y_true, scores, threshold):
    predictions = scores > threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[False, True]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "attack_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("Device:", DEVICE)
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["Label"])

    benign = df[df["Label"].str.upper() == "BENIGN"].copy()
    attacks = df[df["Label"].str.upper() != "BENIGN"].copy()

    # Use the same benign calibration and test records for both models.
    benign = benign.sample(frac=1, random_state=RANDOM_STATE)
    split = int(len(benign) * 0.5)
    calibration = benign.iloc[:split].copy()
    benign_test = benign.iloc[split:].copy()

    # Limit each attack label so the evaluation remains manageable.
    attack_parts = []
    for label, group in attacks.groupby("Label"):
        attack_parts.append(
            group.sample(
                n=min(len(group), 5000),
                random_state=RANDOM_STATE,
            )
        )

    attack_test = pd.concat(attack_parts, ignore_index=True)

    test = pd.concat([benign_test, attack_test], ignore_index=True)
    y_test = test["Label"].str.upper().ne("BENIGN").to_numpy()

    print("Benign calibration records:", len(calibration))
    print("Benign test records:", len(benign_test))
    print("Attack test records:", len(attack_test))
    print("Total test records:", len(test))

    # Load both models and their corresponding scalers.
    production_model = load_model(ROOT / "models" / "autoencoder.pth")
    candidate_model = load_model(
        ROOT / "models" / "candidate_autoencoder.pth"
    )

    import joblib

    production_scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    candidate_scaler = joblib.load(ROOT / "models" / "candidate_scaler.pkl")

    results = {}

    for name, model, scaler in [
        ("production", production_model, production_scaler),
        ("candidate", candidate_model, candidate_scaler),
    ]:
        print(f"\nScoring {name} model...")

        calibration_scores = calculate_scores(model, scaler, calibration)
        benign_test_scores = calculate_scores(model, scaler, benign_test)
        attack_test_scores = calculate_scores(model, scaler, attack_test)

        # Calibrate each model independently on the same benign records.
        threshold = float(np.quantile(
            calibration_scores, 1.0 - TARGET_FPR
        ))

        test_scores = np.concatenate([
            benign_test_scores,
            attack_test_scores,
        ])

        metrics = evaluate(y_test, test_scores, threshold)

        metrics["target_calibration_fpr"] = TARGET_FPR
        metrics["measured_calibration_fpr"] = float(
            np.mean(calibration_scores > threshold)
        )
        metrics["benign_test_false_positive_rate"] = float(
            np.mean(benign_test_scores > threshold)
        )
        metrics["attack_test_recall"] = float(
            np.mean(attack_test_scores > threshold)
        )

        results[name] = metrics

        print(json.dumps(metrics, indent=2))

    report = {
        "dataset": "Wednesday-workingHours.pcap_ISCX.csv",
        "device": str(DEVICE),
        "target_false_positive_rate": TARGET_FPR,
        "calibration_records": int(len(calibration)),
        "benign_test_records": int(len(benign_test)),
        "attack_test_records": int(len(attack_test)),
        "results": results,
        "limitations": [
            "Both models use the same calibration and test records.",
            "The dataset is from the same source used in earlier experiments.",
            "Some records may have been seen during previous model training.",
            "This comparison is not proof of performance on unseen real-world traffic.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nComparison saved to:", REPORT_PATH)
    print("\nDo not replace production model files based on this test alone.")


if __name__ == "__main__":
    main()
