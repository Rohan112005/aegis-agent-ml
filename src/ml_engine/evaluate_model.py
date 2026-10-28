
"""Evaluate AegisAgent and compare current and candidate thresholds."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)


# Project root: aegis-agent-ml
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
CANDIDATE_PATH = ROOT / "reports" / "candidate_threshold.json"
RESULTS_PATH = ROOT / "reports" / "evaluation_report.json"

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

SEED = 42
MAX_PER_CLASS = 10000
BATCH_SIZE = 1024


def load_artifacts():
    required_paths = [
        DATA_PATH,
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
        CANDIDATE_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    scaler = joblib.load(SCALER_PATH)

    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold_data = json.load(file)

    with CANDIDATE_PATH.open(encoding="utf-8") as file:
        candidate_data = json.load(file)

    saved_threshold = float(threshold_data["threshold"])
    candidate_threshold = float(
        candidate_data["candidate_threshold"]
    )

    for name, value in [
        ("Current threshold", saved_threshold),
        ("Candidate threshold", candidate_threshold),
    ]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be finite and non-negative."
            )

    model = PacketAutoencoder(input_dim=len(FEATURES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    if getattr(scaler, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError(
            "Saved scaler has an unexpected feature count."
        )

    return model, scaler, saved_threshold, candidate_threshold


def load_evaluation_data():
    print("Loading CICIDS2017 dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [
        column for column in FEATURES + ["Label"]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    df["Label"] = df["Label"].astype(str).str.strip()

    df = df.dropna(
        subset=FEATURES + ["Label"]
    ).copy()

    df = df[
        (df[FEATURES] >= 0).all(axis=1)
    ].copy()

    benign = df[
        df["Label"].str.upper() == "BENIGN"
    ].copy()

    attacks = df[
        df["Label"].str.upper() != "BENIGN"
    ].copy()

    if benign.empty:
        raise ValueError("No benign records found.")

    if attacks.empty:
        raise ValueError("No attack records found.")

    # Reproduce the benign split used by the existing evaluation.
    _, benign_validation = train_test_split(
        benign,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    samples = [benign_validation]

    # Sample attack records to keep evaluation manageable.
    for _, group in attacks.groupby("Label", sort=True):
        samples.append(
            group.sample(
                n=min(len(group), MAX_PER_CLASS),
                random_state=SEED,
            )
        )

    result = pd.concat(
        samples,
        ignore_index=True,
    )

    return result.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)


def calculate_scores(model, scaler, df):
    scaled = scaler.transform(df[FEATURES])

    if not np.isfinite(scaled).all():
        raise ValueError(
            "Scaled features contain NaN or infinity."
        )

    scores = []

    with torch.no_grad():
        for start in range(0, len(scaled), BATCH_SIZE):
            batch = torch.tensor(
                scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
            )

            batch_scores = compute_reconstruction_loss(
                model,
                batch,
            )

            scores.extend(
                batch_scores.cpu().numpy().tolist()
            )

    scores = np.asarray(scores, dtype=np.float64)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Reconstruction scores contain invalid values."
        )

    return scores


def metrics_at_threshold(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "false_positive_rate": (
            float(fp / (fp + tn))
            if (fp + tn)
            else None
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def print_comparison(current, candidate):
    metrics = [
        ("Threshold", "threshold"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("False-positive rate", "false_positive_rate"),
        ("True negatives", "true_negative"),
        ("False positives", "false_positive"),
        ("False negatives", "false_negative"),
        ("True positives", "true_positive"),
    ]

    print("\n=== Current vs Candidate Threshold ===")
    print(
        f"{'Metric':<24}"
        f"{'Current':>18}"
        f"{'Candidate':>18}"
    )
    print("-" * 60)

    for label, key in metrics:
        current_value = current[key]
        candidate_value = candidate[key]

        if key == "false_positive_rate":
            current_text = (
                f"{current_value:.2%}"
                if current_value is not None
                else "N/A"
            )
            candidate_text = (
                f"{candidate_value:.2%}"
                if candidate_value is not None
                else "N/A"
            )
        elif key in {
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        }:
            current_text = str(current_value)
            candidate_text = str(candidate_value)
        elif key == "threshold":
            current_text = f"{current_value:.10g}"
            candidate_text = f"{candidate_value:.10g}"
        else:
            current_text = f"{current_value:.4f}"
            candidate_text = f"{candidate_value:.4f}"

        print(
            f"{label:<24}"
            f"{current_text:>18}"
            f"{candidate_text:>18}"
        )


def main():
    (
        model,
        scaler,
        saved_threshold,
        candidate_threshold,
    ) = load_artifacts()

    df = load_evaluation_data()

    y_true = (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int).to_numpy()

    scores = calculate_scores(
        model,
        scaler,
        df,
    )

    current_result = metrics_at_threshold(
        y_true,
        scores,
        saved_threshold,
    )

    candidate_result = metrics_at_threshold(
        y_true,
        scores,
        candidate_threshold,
    )

    print("\n=== AegisAgent Threshold Evaluation ===")
    print(f"Records evaluated: {len(df):,}")
    print(f"Benign records: {(y_true == 0).sum():,}")
    print(f"Attack records: {(y_true == 1).sum():,}")

    print_comparison(
        current_result,
        candidate_result,
    )

    print("\n=== Current Threshold Classification Report ===")
    print(
        classification_report(
            y_true,
            (scores > saved_threshold).astype(int),
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
            digits=4,
        )
    )

    print("\n=== Candidate Threshold Classification Report ===")
    print(
        classification_report(
            y_true,
            (scores > candidate_threshold).astype(int),
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
            digits=4,
        )
    )

    # Calculate false-positive rates using benign records only.
    benign_mask = y_true == 0
    benign_scores = scores[benign_mask]

    current_benign_fpr = float(
        np.mean(benign_scores > saved_threshold)
    )

    candidate_benign_fpr = float(
        np.mean(benign_scores > candidate_threshold)
    )

    print("\n=== Benign-Only False-Positive Rates ===")
    print(f"Current threshold:   {current_benign_fpr:.2%}")
    print(f"Candidate threshold: {candidate_benign_fpr:.2%}")

    report = {
        "evaluation_type": (
            "CICIDS2017 Wednesday dataset; benign validation "
            "split and sampled attack records"
        ),
        "records_evaluated": int(len(df)),
        "benign_records": int((y_true == 0).sum()),
        "attack_records": int((y_true == 1).sum()),
        "saved_threshold": saved_threshold,
        "candidate_threshold": candidate_threshold,
        "current_metrics": current_result,
        "candidate_metrics": candidate_result,
        "benign_only_false_positive_rates": {
            "current": current_benign_fpr,
            "candidate": candidate_benign_fpr,
        },
        "warning": (
            "Attack records are sampled from CICIDS2017 Wednesday. "
            "This comparison is exploratory, not independent validation. "
            "The benign validation split may already have been used for "
            "threshold calibration. Do not use these results alone to "
            "claim real-world performance."
        ),
        "configuration_changed": False,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nDetailed report saved to: {RESULTS_PATH}")
    print(
        "The model, scaler, and existing threshold were not changed."
    )


if __name__ == "__main__":
    main()
