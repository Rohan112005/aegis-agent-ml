
"""Train and evaluate a candidate AegisAgent autoencoder.

The existing production model, scaler, and threshold are never overwritten.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)
from src.ml_engine.detector import AnomalyDetector


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CANDIDATE_MODEL = MODELS_DIR / "candidate_autoencoder.pth"
CANDIDATE_SCALER = MODELS_DIR / "candidate_scaler.pkl"
CANDIDATE_THRESHOLD = MODELS_DIR / "candidate_threshold.json"
REPORT_PATH = REPORTS_DIR / "candidate_model_evaluation.json"

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
MAX_TRAIN_ROWS = 100_000
MAX_TEST_PER_ATTACK = 5_000
MAX_BENIGN_TEST = 20_000
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 95.0

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    print("Loading CICIDS2017 Wednesday dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df["Label"] = df["Label"].astype(str).str.strip()

    df = df.dropna(subset=required)
    df = df[(df[FEATURES] >= 0).all(axis=1)].copy()

    benign = df[df["Label"].str.upper() == "BENIGN"].copy()
    attacks = df[df["Label"].str.upper() != "BENIGN"].copy()

    if len(benign) < 10 or attacks.empty:
        raise ValueError("Insufficient benign or attack records.")

    return benign, attacks


def calculate_scores(model, scaler, values):
    scaled = scaler.transform(
        np.asarray(values, dtype=np.float64)
    )

    if not np.isfinite(scaled).all():
        raise ValueError("Scaled features contain invalid values.")

    model.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(scaled), BATCH_SIZE):
            batch = torch.as_tensor(
                scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )
            loss = compute_reconstruction_loss(model, batch)
            scores.extend(
                loss.detach().cpu().numpy().reshape(-1).tolist()
            )

    scores = np.asarray(scores, dtype=np.float64)

    if not np.isfinite(scores).all():
        raise ValueError("Model produced invalid reconstruction scores.")

    return scores


def metrics_at_threshold(y_true, scores, threshold):
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    benign, attacks = load_data()

    # Split benign data into independent training, calibration and test sets.
    benign_train, remaining = train_test_split(
        benign, test_size=0.30, random_state=SEED
    )
    benign_calibration, benign_test = train_test_split(
        remaining, test_size=0.50, random_state=SEED
    )

    # Sample only from the training partition. Calibration and test rows
    # are kept separate from training.
    if len(benign_train) > MAX_TRAIN_ROWS:
        benign_train = benign_train.sample(
            n=MAX_TRAIN_ROWS, random_state=SEED
        )

    if len(benign_calibration) > MAX_TRAIN_ROWS:
        benign_calibration = benign_calibration.sample(
            n=MAX_TRAIN_ROWS, random_state=SEED
        )

    if len(benign_test) > MAX_BENIGN_TEST:
        benign_test = benign_test.sample(
            n=MAX_BENIGN_TEST, random_state=SEED
        )

    attack_test_parts = []
    for _, group in attacks.groupby("Label", sort=True):
        attack_test_parts.append(
            group.sample(
                n=min(len(group), MAX_TEST_PER_ATTACK),
                random_state=SEED,
            )
        )

    attack_test = pd.concat(attack_test_parts, ignore_index=True)

    print("Training benign records:", len(benign_train))
    print("Calibration benign records:", len(benign_calibration))
    print("Test benign records:", len(benign_test))
    print("Test attack records:", len(attack_test))
    print("Compute device:", DEVICE)

    # Fit the candidate scaler on training data only.
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(benign_train[FEATURES])
    x_calibration = benign_calibration[FEATURES].to_numpy(dtype=float)

    train_dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32)
    )
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = PacketAutoencoder(input_dim=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    print("\nTraining candidate autoencoder...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        rows_seen = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            per_sample_loss = compute_reconstruction_loss(model, batch)
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            rows_seen += len(batch)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- loss: {total_loss / rows_seen:.8f}"
        )

    # Choose the threshold using benign calibration data only.
    calibration_scores = calculate_scores(
        model, scaler, x_calibration
    )
    threshold = float(
        np.percentile(calibration_scores, THRESHOLD_PERCENTILE)
    )

    # Test the candidate on unseen benign and labelled attack rows.
    test_df = pd.concat(
        [benign_test, attack_test], ignore_index=True
    )
    y_true = (
        test_df["Label"].str.upper() != "BENIGN"
    ).astype(int).to_numpy()
    x_test = test_df[FEATURES].to_numpy(dtype=float)

    candidate_scores = calculate_scores(model, scaler, x_test)
    candidate_metrics = metrics_at_threshold(
        y_true, candidate_scores, threshold
    )

    # Evaluate the current production model on the exact same test rows.
    current = AnomalyDetector()
    current_scores = calculate_scores(
        current.model,
        current.scaler,
        x_test,
    )
    current_metrics = metrics_at_threshold(
        y_true, current_scores, current.threshold
    )

    # Save candidate artifacts separately. Production files are untouched.
    torch.save(model.cpu().state_dict(), CANDIDATE_MODEL)
    joblib.dump(scaler, CANDIDATE_SCALER)

    threshold_data = {
        "threshold": threshold,
        "percentile": THRESHOLD_PERCENTILE,
        "calibration_samples": len(benign_calibration),
        "input_dim": len(FEATURES),
        "features": FEATURES,
        "training_data": "CICIDS2017 Wednesday BENIGN training split",
        "candidate_only": True,
    }
    CANDIDATE_THRESHOLD.write_text(
        json.dumps(threshold_data, indent=2),
        encoding="utf-8",
    )

    try:
        auc = float(roc_auc_score(y_true, candidate_scores))
    except ValueError:
        auc = None

    report = {
        "evaluation_type": (
            "Random held-out rows from CICIDS2017 Wednesday; "
            "not independent of the source dataset"
        ),
        "device": str(DEVICE),
        "features": FEATURES,
        "training_rows": len(benign_train),
        "calibration_rows": len(benign_calibration),
        "test_benign_rows": len(benign_test),
        "test_attack_rows": len(attack_test),
        "candidate_threshold": threshold,
        "candidate_roc_auc": auc,
        "current_model_metrics": current_metrics,
        "candidate_model_metrics": candidate_metrics,
        "candidate_artifacts": {
            "model": str(CANDIDATE_MODEL),
            "scaler": str(CANDIDATE_SCALER),
            "threshold": str(CANDIDATE_THRESHOLD),
        },
        "production_artifacts_changed": False,
        "warning": (
            "This is a held-out row evaluation from the same source "
            "dataset, not a test on an independent capture or day. "
            "Do not deploy the candidate based on this result alone."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Current vs Candidate Model ===")
    print(json.dumps({
        "current_model": current_metrics,
        "candidate_model": candidate_metrics,
        "candidate_roc_auc": auc,
    }, indent=2))

    print("\nCandidate model saved to:", CANDIDATE_MODEL)
    print("Candidate scaler saved to:", CANDIDATE_SCALER)
    print("Candidate threshold saved to:", CANDIDATE_THRESHOLD)
    print("Evaluation report saved to:", REPORT_PATH)
    print("\nProduction model and threshold were not changed.")


if __name__ == "__main__":
    main()
