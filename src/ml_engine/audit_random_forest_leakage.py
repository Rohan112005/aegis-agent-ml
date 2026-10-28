
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
REPORT_PATH = ROOT / "reports" / "random_forest_leakage_audit.json"

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

MAX_BENIGN = 50000
MAX_PER_ATTACK = 10000
RANDOM_STATE = 42
TEST_SIZE = 0.25


def feature_hashes(frame):
    """Create stable signatures from the ten original feature values."""
    return pd.util.hash_pandas_object(
        frame[FEATURES],
        index=False,
    ).astype("uint64")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [
        col for col in FEATURES + ["Label"]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Target"] = (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)

    # Audit exact feature signatures in the cleaned full dataset.
    print("Checking duplicate feature vectors in full dataset...")
    full_hashes = feature_hashes(df)
    full = df.copy()
    full["_hash"] = full_hashes.to_numpy()

    duplicated_mask = full["_hash"].duplicated(keep=False)
    duplicate_rows = full[duplicated_mask]

    unique_labels_per_hash = (
        full.groupby("_hash")["Label"].nunique()
    )
    conflicting_hashes = set(
        unique_labels_per_hash[unique_labels_per_hash > 1].index
    )

    conflicting_rows = full[
        full["_hash"].isin(conflicting_hashes)
    ]

    # Reproduce the candidate's sampling procedure exactly.
    benign = df[df["Label"].str.upper() == "BENIGN"]
    benign = benign.sample(
        n=min(len(benign), MAX_BENIGN),
        random_state=RANDOM_STATE,
    )

    attack_groups = []
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    for label, group in attacks.groupby("Label"):
        attack_groups.append(
            group.sample(
                n=min(len(group), MAX_PER_ATTACK),
                random_state=RANDOM_STATE,
            )
        )

    sampled = pd.concat(
        [benign] + attack_groups,
        ignore_index=True,
    )

    X = sampled[FEATURES]
    y = sampled["Target"]
    labels = sampled["Label"]

    X_train, X_test, y_train, y_test, labels_train, labels_test = (
        train_test_split(
            X,
            y,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    )

    train_hash = feature_hashes(X_train)
    test_hash = feature_hashes(X_test)

    train_hash_set = set(train_hash.tolist())
    test_hash_set = set(test_hash.tolist())
    overlapping_hashes = train_hash_set.intersection(test_hash_set)

    # Count test rows whose exact feature signature appears in training.
    test_overlap_mask = test_hash.isin(train_hash_set)
    overlap_test_rows = int(test_overlap_mask.sum())

    train_signatures = pd.DataFrame({
        "hash": train_hash.to_numpy(),
        "label": labels_train.to_numpy(),
    })

    test_signatures = pd.DataFrame({
        "hash": test_hash.to_numpy(),
        "label": labels_test.to_numpy(),
    })

    train_labels_by_hash = train_signatures.groupby("hash")["label"].agg(
        lambda values: set(values)
    )

    conflicting_overlap_hashes = []
    for signature in overlapping_hashes:
        test_label_set = set(
            test_signatures.loc[
                test_signatures["hash"] == signature, "label"
            ]
        )
        train_label_set = train_labels_by_hash[signature]

        if train_label_set != test_label_set or len(train_label_set) > 1:
            conflicting_overlap_hashes.append(int(signature))

    overlap_by_label = {}
    test_labels_array = labels_test.to_numpy()

    for label in sorted(set(test_labels_array)):
        mask = test_labels_array == label
        overlap_by_label[label] = {
            "test_records": int(mask.sum()),
            "exact_feature_overlap_records": int(
                test_overlap_mask.to_numpy()[mask].sum()
            ),
        }

    report = {
        "dataset": DATA_PATH.name,
        "sampling_settings": {
            "max_benign": MAX_BENIGN,
            "max_per_attack": MAX_PER_ATTACK,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        "full_cleaned_dataset": {
            "records": int(len(df)),
            "rows_with_repeated_feature_vectors": int(
                duplicated_mask.sum()
            ),
            "unique_repeated_feature_vectors": int(
                full.loc[duplicated_mask, "_hash"].nunique()
            ),
            "feature_vectors_with_multiple_labels": int(
                len(conflicting_hashes)
            ),
            "rows_in_conflicting_label_groups": int(
                len(conflicting_rows)
            ),
        },
        "candidate_train_test_split": {
            "training_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "unique_training_feature_vectors": int(
                train_hash.nunique()
            ),
            "unique_test_feature_vectors": int(
                test_hash.nunique()
            ),
            "unique_feature_vectors_in_both_sets": int(
                len(overlapping_hashes)
            ),
            "test_rows_whose_features_appear_in_training": (
                overlap_test_rows
            ),
            "test_row_overlap_percent": round(
                100 * overlap_test_rows / len(X_test), 4
            ),
            "overlap_signatures_with_label_set_differences": int(
                len(conflicting_overlap_hashes)
            ),
            "overlap_by_label": overlap_by_label,
        },
        "limitations": [
            "This checks exact feature-vector duplicates, not near duplicates.",
            "Feature signatures use pandas 64-bit hashes, so theoretical "
            "hash collisions are possible.",
            "Repeated feature vectors can be legitimate and do not by "
            "themselves prove leakage.",
            "The candidate's test split is not independent of the source "
            "dataset or necessarily of previous experiments.",
            "This audit does not prove real-world generalization.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Full Dataset Duplicate Audit ===")
    print("Cleaned records:", len(df))
    print(
        "Rows with repeated feature vectors:",
        report["full_cleaned_dataset"][
            "rows_with_repeated_feature_vectors"
        ],
    )
    print(
        "Feature vectors associated with multiple labels:",
        report["full_cleaned_dataset"][
            "feature_vectors_with_multiple_labels"
        ],
    )

    print("\n=== Candidate Train/Test Overlap ===")
    print("Training records:", len(X_train))
    print("Test records:", len(X_test))
    print(
        "Unique feature vectors in both sets:",
        report["candidate_train_test_split"][
            "unique_feature_vectors_in_both_sets"
        ],
    )
    print(
        "Test rows repeated in training:",
        overlap_test_rows,
    )
    print(
        "Test-row overlap percentage:",
        report["candidate_train_test_split"][
            "test_row_overlap_percent"
        ],
        "%",
    )
    print(
        "Overlap signatures with label-set differences:",
        len(conflicting_overlap_hashes),
    )

    print("\n=== Exact Overlap by Test Label ===")
    for label, stats in overlap_by_label.items():
        print(
            f"{label}: {stats['exact_feature_overlap_records']} / "
            f"{stats['test_records']} test rows overlap with training"
        )

    print("\nReport saved:", REPORT_PATH)
    print("No model or threshold files were changed.")


if __name__ == "__main__":
    main()
