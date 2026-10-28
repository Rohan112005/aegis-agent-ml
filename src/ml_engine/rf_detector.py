
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "candidate_random_forest_group_eval.joblib"

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


class RandomForestDetector:
    """Experimental Random Forest detector for binary flow classification."""

    def __init__(self, model_path=MODEL_PATH, threshold=0.50):
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.model_loaded = False

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Candidate model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        # Confirm the model expects the same feature names and order.
        model_features = getattr(self.model, "feature_names_in_", None)

        if model_features is None:
            raise ValueError(
                "The model does not contain feature names. "
                "Check how the candidate model was trained."
            )

        if list(model_features) != FEATURES:
            raise ValueError(
                "Candidate model feature names or order do not match FEATURES."
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "The candidate model must support predict_proba()."
            )

        classes = list(self.model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The candidate model does not contain attack class 1."
            )

        self.attack_class_index = classes.index(1)
        self.model_loaded = True

    def predict(self, values):
        """
        Predict whether one flow is anomalous.

        Accepts:
          - a dictionary containing the ten named features, or
          - a list, tuple, or NumPy array in FEATURES order.

        Class 1 represents attack traffic; class 0 represents benign traffic.
        """

        if isinstance(values, dict):
            missing = [name for name in FEATURES if name not in values]

            if missing:
                raise ValueError(
                    f"Missing required features: {missing}"
                )

            frame = pd.DataFrame(
                [[values[name] for name in FEATURES]],
                columns=FEATURES,
            )

        else:
            try:
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Features must be numeric."
                ) from exc

            if array.ndim == 1:
                array = array.reshape(1, -1)

            if array.shape != (1, len(FEATURES)):
                raise ValueError(
                    f"Expected one flow with {len(FEATURES)} features."
                )

            frame = pd.DataFrame(array, columns=FEATURES)

        try:
            frame = frame.apply(pd.to_numeric, errors="raise")
            array = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All flow features must be numeric."
            ) from exc

        if not np.isfinite(array).all():
            raise ValueError(
                "Features must contain only finite numbers."
            )

        if (array < 0).any():
            raise ValueError(
                "Flow features cannot contain negative values."
            )

        probabilities = self.model.predict_proba(frame)[0]
        score = float(probabilities[self.attack_class_index])
        anomaly = score >= self.threshold

        return {
            "anomaly": bool(anomaly),
            "score": score,
            "threshold": self.threshold,
            "status": "ANOMALY" if anomaly else "NORMAL",
            "model": "random_forest_candidate",
            "model_loaded": self.model_loaded,
        }
