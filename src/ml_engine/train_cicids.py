
"""
Train AegisAgent Autoencoder using CICIDS2017 Wednesday traffic.

Training:
- Uses BENIGN traffic only.
- Cleans NaN and infinite values.
- Selects 10 network-flow features.
- Fits and saves a StandardScaler.
- Calculates an anomaly threshold from normal validation data.
- Saves the model, scaler, and threshold.
"""

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

INPUT_DIM = 10
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.001

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Selected CICIDS2017 features
# --------------------------------------------------

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


# --------------------------------------------------
# Load and clean dataset
# --------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading CICIDS2017 dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Remove spaces around CSV column names.
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("The CSV does not contain a Label column.")

    # Confirm all selected columns exist.
    missing_columns = [
        col for col in FEATURES if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    # Convert selected features to numerical values.
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinite values with NaN.
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Keep rows where all selected features are valid.
    valid_rows = df[FEATURES].notna().all(axis=1)

    df = df.loc[valid_rows].copy()

    # Remove rows with negative flow statistics.
    non_negative_features = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
    ]

    valid_non_negative = (
        df[non_negative_features] >= 0
    ).all(axis=1)

    df = df.loc[valid_non_negative].copy()

    # Remove accidental leading/trailing spaces from labels.
    df["Label"] = df["Label"].astype(str).str.strip()

    print(f"Rows after cleaning: {len(df)}")
    print("\nLabels after cleaning:")
    print(df["Label"].value_counts())

    return df


# --------------------------------------------------
# Train Autoencoder
# --------------------------------------------------

def train_model():
    df = load_dataset()

    # Use only normal traffic for Autoencoder training.
    benign = df[df["Label"] == "BENIGN"].copy()

    if len(benign) < 100:
        raise ValueError(
            "Not enough BENIGN records to train the model."
        )

    print(
        f"\nNormal BENIGN records available: {len(benign)}"
    )

    # Split normal traffic into training and validation sets.
    train_df, validation_df = train_test_split(
        benign,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    # Fit a [0, 1] scaler only on the training data.
    # This matches the autoencoder's Sigmoid output layer.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    train_scaled = scaler.fit_transform(
        train_df[FEATURES]
    )

    validation_scaled = scaler.transform(
        validation_df[FEATURES]
    )

    # Convert data to PyTorch tensors.
    train_tensor = torch.tensor(
        train_scaled,
        dtype=torch.float32,
    )

    validation_tensor = torch.tensor(
        validation_scaled,
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Create the Autoencoder.
    model = PacketAutoencoder(input_dim=INPUT_DIM)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Mean Squared Error training loss.
    criterion = nn.MSELoss()

    print("\nStarting Autoencoder training...")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for (batch,) in train_loader:
            optimizer.zero_grad()

            reconstructed = model(batch)

            loss = criterion(
                reconstructed,
                batch,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_samples += len(batch)

        average_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Training Loss: {average_loss:.6f}"
        )

    # --------------------------------------------------
    # Calculate threshold using normal validation data
    # --------------------------------------------------

    model.eval()

    validation_scores = []

    with torch.no_grad():
        for start in range(
            0,
            len(validation_tensor),
            BATCH_SIZE,
        ):
            batch = validation_tensor[
                start:start + BATCH_SIZE
            ]

            scores = compute_reconstruction_loss(
                model,
                batch,
            )

            validation_scores.extend(
                scores.cpu().numpy().tolist()
            )

    # 95th percentile: 5% of normal validation scores
    # are above this threshold.
    threshold = float(
        np.percentile(validation_scores, 95)
    )

    print("\nTraining completed.")
    print(f"Validation samples: {len(validation_scores)}")
    print(f"Calculated threshold: {threshold:.8f}")

    # --------------------------------------------------
    # Save model, scaler, and threshold
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    threshold_data = {
        "threshold": threshold,
        "percentile": 95.0,
        "samples": len(validation_scores),
        "input_dim": INPUT_DIM,
        "training_data": "CICIDS2017 Wednesday BENIGN traffic",
        "features": FEATURES,
    }

    with THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_data,
            file,
            indent=4,
        )

    print("\nSaved files:")
    print(f"Model:     {MODEL_PATH}")
    print(f"Scaler:    {SCALER_PATH}")
    print(f"Threshold: {THRESHOLD_PATH}")


if __name__ == "__main__":
    train_model()
