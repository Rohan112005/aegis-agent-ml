
"""Explore candidate threshold trade-offs without changing artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "candidate_autoencoder.pth"
SCALER_PATH = ROOT / "models" / "candidate_scaler.pkl"
THRESHOLD_PATH = ROOT / "models" / "candidate_threshold.json"

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
BATCH_SIZE = 1024


def score_rows(model, scaler, values):
    scaled = scaler.transform(
        pd.DataFrame(values, columns=FEATURES)
    )
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(scaled), BATCH_SIZE):
            batch = torch.as_tensor(
                scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
            )
            loss = compute_reconstruction_loss(model, batch)
            scores.extend(loss.cpu().numpy().reshape(-1).tolist())

    return np.asarray(scores)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + ["Label"]
    )
    df = df[(df[FEATURES] >= 0).all(axis=1)]

    benign = df[df["Label"].str.upper() == "BENIGN"]
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    # Keep a separate benign test subset and a stratified attack test subset.
    _, benign_test = train_test_split(
        benign, test_size=0.20, random_state=123
    )
    attack_parts = [
        group.sample(
            n=min(len(group), 5000), random_state=123
        )
        for _, group in attacks.groupby("Label", sort=True)
    ]
    attack_test = pd.concat(attack_parts, ignore_index=True)

    test = pd.concat(
        [benign_test, attack_test], ignore_index=True
    )
    y = (
        test["Label"].str.upper() != "BENIGN"
    ).to_numpy(dtype=int)

    scaler = joblib.load(SCALER_PATH)
    threshold_data = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    model = PacketAutoencoder(input_dim=len(FEATURES))
    state = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    scores = score_rows(
        model, scaler, test[FEATURES].to_numpy(dtype=float)
    )

    # The existing calibration threshold is included for comparison.
    thresholds = [
        threshold_data["threshold"] * factor
        for factor in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    ]

    print("Threshold | Attack recall | Benign false-positive rate")
    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    for threshold in thresholds:
        recall = np.mean(attack_scores > threshold)
        fpr = np.mean(benign_scores > threshold)
        print(f"{threshold:.10g} | {recall:.2%} | {fpr:.2%}")

    print("\nThis is exploratory threshold analysis, not independent validation.")
    print("No model, scaler, or threshold files were modified.")


if __name__ == "__main__":
    main()
