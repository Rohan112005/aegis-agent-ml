
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.ml_engine.autoencoder import PacketAutoencoder


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

REPORT_DIR = ROOT / "reports"
FEATURE_REPORT = REPORT_DIR / "feature_distributions.csv"
ERROR_REPORT = REPORT_DIR / "reconstruction_errors_by_attack.csv"
SUMMARY_REPORT = REPORT_DIR / "attack_feature_analysis.json"

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

# Limit dataset size to keep CPU memory and runtime manageable.
MAX_BENIGN = 20000
MAX_PER_ATTACK = 10000
BATCH_SIZE = 2048
RANDOM_STATE = 2026

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Load production threshold without modifying it
# ---------------------------------------------------------

def load_threshold():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "threshold" not in data:
        raise ValueError(
            "The production threshold file does not contain "
            "the expected 'threshold' field."
        )

    return float(data["threshold"])


# ---------------------------------------------------------
# Calculate total and per-feature reconstruction errors
# ---------------------------------------------------------

def calculate_reconstruction_errors(model, scaler, frame):
    raw_values = frame[FEATURES].astype(np.float64)
    scaled_values = scaler.transform(raw_values)

    all_scores = []
    all_feature_errors = []

    with torch.no_grad():
        for start in range(0, len(scaled_values), BATCH_SIZE):
            batch_array = scaled_values[start:start + BATCH_SIZE]

            batch = torch.as_tensor(
                batch_array,
                dtype=torch.float32,
                device=DEVICE,
            )

            reconstructed = model(batch)

            # Squared reconstruction error for each feature.
            feature_errors = (reconstructed - batch) ** 2

            # Match the model's average-per-feature reconstruction score.
            scores = feature_errors.mean(dim=1)

            all_scores.append(scores.cpu().numpy())
            all_feature_errors.append(feature_errors.cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_feature_errors),
    )


# ---------------------------------------------------------
# Summarize raw feature distributions
# ---------------------------------------------------------

def build_feature_distribution_report(df):
    rows = []

    for label, group in df.groupby("Label", sort=True):
        for feature in FEATURES:
            values = group[feature].to_numpy(dtype=np.float64)

            rows.append({
                "label": str(label),
                "feature": feature,
                "records": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Summarize reconstruction errors by attack type
# ---------------------------------------------------------

def build_error_report(df, feature_errors, threshold):
    rows = []

    for label in sorted(df["Label"].unique()):
        mask = df["Label"].to_numpy() == label
        label_frame = df.loc[mask]

        errors = feature_errors[mask]
        scores = label_frame["reconstruction_score"].to_numpy()
        detected = scores > threshold

        for index, feature in enumerate(FEATURES):
            feature_error = errors[:, index]

            rows.append({
                "label": str(label),
                "feature": feature,
                "records": int(len(label_frame)),
                "mean_squared_error": float(np.mean(feature_error)),
                "median_squared_error": float(np.median(feature_error)),
                "p95_squared_error": float(
                    np.percentile(feature_error, 95)
                ),
                "detected_mean_squared_error": (
                    float(np.mean(feature_error[detected]))
                    if np.any(detected) else None
                ),
                "missed_mean_squared_error": (
                    float(np.mean(feature_error[~detected]))
                    if np.any(~detected) else None
                ),
            })

    return pd.DataFrame(rows)


def build_detection_summary(df, threshold):
    rows = []

    for label, group in df.groupby("Label", sort=True):
        scores = group["reconstruction_score"].to_numpy()
        detected = scores > threshold

        total = len(group)
        count_detected = int(detected.sum())

        rows.append({
            "label": str(label),
            "records_tested": int(total),
            "detected": count_detected,
            "missed": int(total - count_detected),
            "recall_percent": round(
                100 * count_detected / total, 2
            ),
            "score_median": float(np.median(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_mean": float(np.mean(scores)),
        })

    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [
        column for column in FEATURES + ["Label"]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    benign = df[df["Label"].str.upper() == "BENIGN"]

    attack_parts = []
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    for label, group in attacks.groupby("Label"):
        attack_parts.append(
            group.sample(
                n=min(len(group), MAX_PER_ATTACK),
                random_state=RANDOM_STATE,
            )
        )

    if len(benign) > MAX_BENIGN:
        benign = benign.sample(
            n=MAX_BENIGN,
            random_state=RANDOM_STATE,
        )

    sampled = pd.concat(
        [benign] + attack_parts,
        ignore_index=True,
    )

    print("Benign records sampled:", len(benign))
    print("Total records sampled:", len(sampled))
    print("Attack categories:", sampled["Label"].nunique() - 1)

    print("\nLoading production model...")
    model = PacketAutoencoder(input_dim=len(FEATURES)).to(DEVICE)

    try:
        state = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    threshold = load_threshold()

    print(f"Production threshold: {threshold:.12g}")
    print("\nCalculating reconstruction errors...")

    scores, feature_errors = calculate_reconstruction_errors(
        model, scaler, sampled
    )

    sampled = sampled.reset_index(drop=True)
    sampled["reconstruction_score"] = scores
    sampled["detected"] = scores > threshold

    print("\nBuilding raw feature distribution report...")
    feature_report = build_feature_distribution_report(sampled)
    feature_report.to_csv(FEATURE_REPORT, index=False)

    print("Building per-feature reconstruction error report...")
    error_report = build_error_report(
        sampled, feature_errors, threshold
    )
    error_report.to_csv(ERROR_REPORT, index=False)

    detection_summary = build_detection_summary(sampled, threshold)

    # Rank features by their average reconstruction error for each label.
    top_errors = {}

    for label in sorted(sampled["Label"].unique()):
        label_rows = error_report[
            error_report["label"] == label
        ].sort_values(
            "mean_squared_error",
            ascending=False,
        )

        top_errors[str(label)] = [
            {
                "feature": row["feature"],
                "mean_squared_error": float(
                    row["mean_squared_error"]
                ),
                "median_squared_error": float(
                    row["median_squared_error"]
                ),
            }
            for _, row in label_rows.head(5).iterrows()
        ]

    benign_scores = sampled.loc[
        sampled["Label"].str.upper() == "BENIGN",
        "reconstruction_score",
    ].to_numpy()

    benign_fpr = float(np.mean(benign_scores > threshold))

    report = {
        "dataset": DATA_PATH.name,
        "model": "production",
        "threshold": threshold,
        "device": str(DEVICE),
        "sample_counts": {
            "benign": int(len(benign)),
            "total": int(len(sampled)),
        },
        "benign_false_positive_rate_percent": round(
            benign_fpr * 100, 2
        ),
        "detection_by_label": detection_summary,
        "top_reconstruction_error_features": top_errors,
        "interpretation_notes": [
            "Raw feature distributions describe input values, not causation.",
            "Reconstruction error shows which scaled inputs the model "
            "reconstructs poorly.",
            "High reconstruction error does not prove that a feature "
            "caused an attack or that the flow is malicious.",
            "This is a sampled analysis of the CICIDS2017 Wednesday dataset.",
            "The dataset may overlap with previous model training data.",
            "No production model, scaler, or threshold was modified.",
        ],
    }

    SUMMARY_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Detection Summary (weakest first) ===")
    for row in sorted(
        detection_summary,
        key=lambda item: item["recall_percent"],
    ):
        print(
            f"{row['label'][:30]:30} "
            f"recall={row['recall_percent']:6.2f}% "
            f"detected={row['detected']:6d} "
            f"missed={row['missed']:6d}"
        )

    print("\n=== Highest Reconstruction-Error Features ===")
    for label, features in top_errors.items():
        print(f"\n{label}:")
        for item in features:
            print(
                f"  {item['feature']}: "
                f"mean squared error="
                f"{item['mean_squared_error']:.8g}"
            )

    print(
        "\nBenign false-positive rate on this sample: "
        f"{benign_fpr * 100:.2f}%"
    )

    print("\nReports saved:")
    print(FEATURE_REPORT)
    print(ERROR_REPORT)
    print(SUMMARY_REPORT)

    print("\nProduction model and threshold remain unchanged.")


if __name__ == "__main__":
    main()
