
"""Anomaly detector wrapper for CICIDS2017-preprocessed traffic."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIM = 10

FEATURE_NAMES = [
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


class AnomalyDetector:
    def __init__(
        self,
        model_path=None,
        threshold_path=None,
        scaler_path=None,
        threshold=None,
    ):
        self.model_path = Path(
            model_path or ROOT / "models" / "autoencoder.pth"
        )
        self.threshold_path = Path(
            threshold_path or ROOT / "models" / "threshold.json"
        )
        self.scaler_path = Path(
            scaler_path or ROOT / "models" / "scaler.pkl"
        )

        self.threshold = (
            float(threshold)
            if threshold is not None
            else self._load_threshold()
        )

        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Threshold must be finite and non-negative.")

        self.scaler = None
        self.model = PacketAutoencoder(input_dim=INPUT_DIM)
        self.model_loaded = False

        if self.model_path.exists() and self.scaler_path.exists():
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.scaler = joblib.load(self.scaler_path)

            if hasattr(self.scaler, "n_features_in_"):
                if self.scaler.n_features_in_ != INPUT_DIM:
                    raise ValueError(
                        "Scaler feature count does not match the model."
                    )

            scaler_names = getattr(
                self.scaler, "feature_names_in_", None
            )
            if scaler_names is not None:
                if list(scaler_names) != FEATURE_NAMES:
                    raise ValueError(
                        "Scaler feature names/order do not match "
                        "the detector's expected feature order."
                    )

            self.model_loaded = True

        self.model.eval()

    def _load_threshold(self):
        """Load the saved threshold."""
        try:
            with self.threshold_path.open(encoding="utf-8") as file:
                threshold_data = json.load(file)

            return float(threshold_data["threshold"])

        except (OSError, ValueError, KeyError, TypeError):
            raise RuntimeError(
                f"Could not load a valid threshold from "
                f"{self.threshold_path}. Train the model first."
            )

    def predict(self, features):
        """Return reconstruction score and input-range diagnostics."""
        try:
            features = np.asarray(
                features, dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Features must contain numeric values."
            ) from exc

        if len(features) != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} features; got {len(features)}"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "Features must contain only finite values."
            )

        if not self.model_loaded or self.scaler is None:
            raise RuntimeError(
                "Model or scaler is missing. "
                "Train the CICIDS2017 model first."
            )

        # Preserve the training feature names and order when available.
        scaler_names = getattr(
            self.scaler, "feature_names_in_", None
        )
        if scaler_names is not None:
            input_data = pd.DataFrame(
                [features], columns=list(scaler_names)
            )
        else:
            input_data = features.reshape(1, -1)

        scaled_features = self.scaler.transform(input_data)

        if not np.isfinite(scaled_features).all():
            raise ValueError(
                "Scaling produced non-finite feature values."
            )

        # MinMaxScaler does not clip by default. Values outside [0, 1]
        # indicate that the input is outside the training range.
        out_of_range = np.logical_or(
            scaled_features[0] < 0.0,
            scaled_features[0] > 1.0,
        )
        out_of_range_features = [
            FEATURE_NAMES[i]
            for i, is_outside in enumerate(out_of_range)
            if is_outside
        ]

        x = torch.tensor(
            scaled_features,
            dtype=torch.float32,
        )

        with torch.no_grad():
            score = float(
                compute_reconstruction_loss(
                    self.model, x
                ).item()
            )

        anomaly = score > self.threshold

        return {
            "anomaly": anomaly,
            "score": score,
            "threshold": self.threshold,
            "status": "ANOMALY" if anomaly else "NORMAL",
            "model_loaded": self.model_loaded,
            "out_of_training_range": bool(out_of_range_features),
            "out_of_range_features": out_of_range_features,
        }
