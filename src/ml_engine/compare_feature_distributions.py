
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
OUTPUT_PATH = ROOT / "reports" / "feature_separation_analysis.csv"

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

MAX_PER_GROUP = 10000
SEED = 2026


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    benign = df[df["Label"].str.upper() == "BENIGN"]
    benign = benign.sample(
        n=min(len(benign), MAX_PER_GROUP),
        random_state=SEED,
    )

    attack_groups = []
    for label, group in df[
        df["Label"].str.upper() != "BENIGN"
    ].groupby("Label"):
        group = group.sample(
            n=min(len(group), MAX_PER_GROUP),
            random_state=SEED,
        )
        attack_groups.append((label, group))

    results = []

    for label, attack in attack_groups:
        print(f"Comparing BENIGN vs {label}...")

        # Same benign sample is reused for every attack comparison.
        for feature in FEATURES:
            benign_values = benign[feature].to_numpy(dtype=float)
            attack_values = attack[feature].to_numpy(dtype=float)

            combined = np.concatenate([benign_values, attack_values])
            labels = np.concatenate([
                np.zeros(len(benign_values), dtype=int),
                np.ones(len(attack_values), dtype=int),
            ])

            # Raw ROC-AUC can be misleading for highly skewed features.
            # Also calculate AUC after log1p for non-negative features.
            raw_auc = float(roc_auc_score(labels, combined))

            if np.min(combined) >= 0:
                transformed = np.log1p(combined)
                log_auc = float(roc_auc_score(labels, transformed))
            else:
                log_auc = None

            results.append({
                "attack_type": label,
                "feature": feature,
                "benign_median": float(np.median(benign_values)),
                "attack_median": float(np.median(attack_values)),
                "benign_p95": float(np.percentile(benign_values, 95)),
                "attack_p95": float(np.percentile(attack_values, 95)),
                "raw_feature_auc": raw_auc,
                "log1p_feature_auc": log_auc,
                "records_benign": int(len(benign_values)),
                "records_attack": int(len(attack_values)),
            })

    output = pd.DataFrame(results)
    output["auc_distance_from_random"] = (
        output["raw_feature_auc"] - 0.5
    ).abs()

    output = output.sort_values(
        ["attack_type", "auc_distance_from_random"],
        ascending=[True, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Top individual features by attack category ===")

    for label in output["attack_type"].unique():
        subset = output[output["attack_type"] == label].head(5)

        print(f"\n{label}")
        print(
            subset[
                [
                    "feature",
                    "benign_median",
                    "attack_median",
                    "raw_feature_auc",
                    "log1p_feature_auc",
                ]
            ].to_string(index=False)
        )

    print("\nSaved report:", OUTPUT_PATH)
    print(
        "\nNote: These are exploratory single-feature comparisons. "
        "They do not establish causation or real-world detection accuracy."
    )


if __name__ == "__main__":
    main()
