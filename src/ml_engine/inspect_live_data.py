
"""Inspect live packet feature data without modifying any files."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LIVE_DATA_PATH = ROOT / "data" / "live_packet_features.csv"

REQUIRED_FEATURES = [
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


def main():
    print("AegisAgent - Live Data Inspection")
    print("=" * 45)

    if not LIVE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Live data file not found: {LIVE_DATA_PATH}"
        )

    df = pd.read_csv(LIVE_DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"\nFile: {LIVE_DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== Available columns ===")
    for column in df.columns:
        print(f"- {column}")

    print("\n=== Required feature check ===")
    missing = [
        feature for feature in REQUIRED_FEATURES
        if feature not in df.columns
    ]

    if missing:
        print("Missing required features:")
        for feature in missing:
            print(f"- {feature}")
    else:
        print("All 10 required model features are present.")

    print("\n=== First five rows ===")
    print(df.head().to_string(index=False))

    if missing:
        print(
            "\nRESULT: This file cannot be passed directly to "
            "the current model. Feature mapping is required."
        )
        return

    features = df[REQUIRED_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_mask = features.notna().all(axis=1)
    nonnegative_mask = (features >= 0).all(axis=1)

    print("\n=== Data quality ===")
    print(f"Rows with all numeric features: {int(valid_mask.sum()):,}")
    print(
        "Rows with all numeric and non-negative features: "
        f"{int((valid_mask & nonnegative_mask).sum()):,}"
    )
    print(
        f"Rows with missing or invalid feature values: "
        f"{int((~valid_mask).sum()):,}"
    )

    print("\n=== Feature summary ===")
    print(features.describe().to_string())

    print("\nInspection complete. No files were modified.")
    print(
        "\nNote: valid feature columns alone do not prove that "
        "this data is independent of training data or represents "
        "real network traffic."
    )


if __name__ == "__main__":
    main()
