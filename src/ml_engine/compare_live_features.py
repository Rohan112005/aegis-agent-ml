"""Compare CICIDS2017 training features and model anomaly scores."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_engine.detector import AnomalyDetector
from src.ml_engine.flow_aggregator import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRAINING_CSV = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
REPORT_PATH = ROOT / "reports" / "live_feature_comparison.json"
LABEL_COLUMN = "Label"
SAMPLE_SIZE = 500
RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    """Load required columns while handling spaces in CSV headers."""
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAINING_CSV}")

    logger.info("Inspecting CICIDS2017 column names...")

    original_columns = pd.read_csv(TRAINING_CSV, nrows=0).columns.tolist()

    # Map cleaned column names to their original names in the CSV.
    column_map = {}
    for original_name in original_columns:
        cleaned_name = original_name.strip()
        if cleaned_name in column_map:
            raise ValueError(
                f"Duplicate column after removing spaces: {cleaned_name!r}"
            )
        column_map[cleaned_name] = original_name

    required_columns = FEATURE_NAMES + [LABEL_COLUMN]
    missing_columns = [
        name for name in required_columns if name not in column_map
    ]

    if missing_columns:
        raise ValueError(
            "Training CSV is missing required columns after whitespace "
            f"normalization: {missing_columns}"
        )

    selected_original_names = [
        column_map[name] for name in required_columns
    ]

    logger.info("Loading the 10 model features and label...")
    frame = pd.read_csv(
        TRAINING_CSV,
        usecols=selected_original_names,
        low_memory=False,
    )

    # Remove leading/trailing spaces from column names.
    frame.columns = frame.columns.str.strip()

    # Convert feature columns to numeric values.
    for name in FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip()

    # CICIDS data can contain infinite rate values or missing values.
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(subset=FEATURE_NAMES + [LABEL_COLUMN], inplace=True)

    logger.info("Usable rows after cleaning: %s", f"{len(frame):,}")
    logger.info("Available labels: %s", frame[LABEL_COLUMN].nunique())

    return frame


def summarize_features(frame: pd.DataFrame) -> dict:
    """Summarize feature distributions for a group of rows."""
    summary = {}

    for name in FEATURE_NAMES:
        values = frame[name].astype(float)

        summary[name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return summary


def score_sample(
    frame: pd.DataFrame,
    detector: AnomalyDetector,
) -> dict:
    """Score a sample using the existing trained detector."""
    if frame.empty:
        return {
            "sample_count": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_percentage": None,
            "score_min": None,
            "score_median": None,
            "score_max": None,
            "error": "No usable rows were available for this group.",
        }

    scores = []
    anomaly_count = 0
    error_count = 0

    for row in frame[FEATURE_NAMES].itertuples(index=False, name=None):
        try:
            prediction = detector.predict([float(value) for value in row])
            scores.append(float(prediction["score"]))

            if bool(prediction["anomaly"]):
                anomaly_count += 1

        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            error_count += 1
            logger.warning("A sample could not be scored: %s", exc)

    if not scores:
        return {
            "sample_count": len(frame),
            "scored_count": 0,
            "error_count": error_count,
            "error": "The detector could not score any rows.",
        }

    score_values = np.asarray(scores, dtype=float)

    return {
        "sample_count": len(frame),
        "scored_count": len(scores),
        "error_count": error_count,
        "anomaly_count": anomaly_count,
        "normal_count": len(scores) - anomaly_count,
        "anomaly_percentage": round(
            100 * anomaly_count / len(scores), 2
        ),
        "score_min": float(np.min(score_values)),
        "score_median": float(np.median(score_values)),
        "score_max": float(np.max(score_values)),
    }


def main() -> None:
    logger.info("Loading CICIDS2017 data...")
    data = load_training_data()

    # CICIDS2017 labels normally use BENIGN for non-attack traffic.
    benign = data[data[LABEL_COLUMN].str.upper() == "BENIGN"]
    attacks = data[data[LABEL_COLUMN].str.upper() != "BENIGN"]

    if benign.empty:
        raise ValueError(
            "No BENIGN rows found. Check the training CSV label values."
        )

    if attacks.empty:
        raise ValueError(
            "No attack rows found. Check the training CSV label values."
        )

    benign_sample = benign.sample(
        n=min(SAMPLE_SIZE, len(benign)),
        random_state=RANDOM_STATE,
    )
    attack_sample = attacks.sample(
        n=min(SAMPLE_SIZE, len(attacks)),
        random_state=RANDOM_STATE,
    )

    logger.info("Loading the existing trained detector...")
    detector = AnomalyDetector()

    if not detector.model_loaded:
        raise RuntimeError(
            "The trained model or scaler could not be loaded. "
            "Check models/autoencoder.pth and models/scaler.pkl."
        )

    report = {
        "training_csv": str(TRAINING_CSV),
        "feature_order": FEATURE_NAMES,
        "threshold": float(detector.threshold),
        "valid_training_rows": int(len(data)),
        "benign_rows_available": int(len(benign)),
        "attack_rows_available": int(len(attacks)),
        "sample_size_per_group": SAMPLE_SIZE,
        "benign_features": summarize_features(benign_sample),
        "attack_features": summarize_features(attack_sample),
        "benign_model_scores": score_sample(benign_sample, detector),
        "attack_model_scores": score_sample(attack_sample, detector),
        "important_note": (
            "These are scores on sampled CICIDS2017 training data, not "
            "scores from live network traffic. This report does not "
            "independently validate the model or select a new threshold."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CICIDS2017 Feature and Score Comparison ===")
    print(f"Valid training rows: {len(data):,}")
    print(f"Benign rows available: {len(benign):,}")
    print(f"Attack rows available: {len(attacks):,}")
    print(f"Model threshold: {detector.threshold:.10g}")

    print("\nBenign sample scores:")
    print(json.dumps(report["benign_model_scores"], indent=2))

    print("\nAttack sample scores:")
    print(json.dumps(report["attack_model_scores"], indent=2))

    print(f"\nFull report saved to:\n{REPORT_PATH}")
    print(
        "\nImportant: This compares sampled training data. "
        "It does not yet compare actual live-flow feature distributions."
    )


if __name__ == "__main__":
    main()