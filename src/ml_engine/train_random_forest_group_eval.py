
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"

MODEL_PATH = ROOT / "models" / "candidate_random_forest_group_eval.joblib"
REPORT_PATH = ROOT / "reports" / "random_forest_group_eval.json"
IMPORTANCE_PATH = ROOT / "reports" / "random_forest_group_feature_importance.csv"

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
N_SPLITS = 4
RANDOM_STATE = 42
PROBABILITY_THRESHOLD = 0.50


def load_and_prepare_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    # Define an exact feature group across the full cleaned dataset.
    # Identical feature vectors receive the same group ID.
    print("Grouping identical feature vectors...")
    df["_group_id"] = (
        df.groupby(FEATURES, sort=False, dropna=False)
        .ngroup()
        .astype("int64")
    )

    # If the same ten features have multiple labels, exclude that
    # entire feature group from this experiment.
    label_counts = df.groupby("_group_id")["Label"].nunique()
    ambiguous_groups = set(
        label_counts[label_counts > 1].index.tolist()
    )

    ambiguous_mask = df["_group_id"].isin(ambiguous_groups)
    ambiguous_rows = int(ambiguous_mask.sum())

    print("Conflicting feature groups:", len(ambiguous_groups))
    print("Rows in conflicting groups excluded:", ambiguous_rows)

    df = df.loc[~ambiguous_mask].copy()

    # Sample benign traffic and each attack label.
    benign = df[df["Label"].str.upper() == "BENIGN"]
    benign = benign.sample(
        n=min(len(benign), MAX_BENIGN),
        random_state=RANDOM_STATE,
    )

    attack_parts = []
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    for label, group in attacks.groupby("Label"):
        attack_parts.append(
            group.sample(
                n=min(len(group), MAX_PER_ATTACK),
                random_state=RANDOM_STATE,
            )
        )

    if not attack_parts:
        raise ValueError("No attack records remain after filtering.")

    sampled = pd.concat([benign] + attack_parts, ignore_index=True)

    sampled["Target"] = (
        sampled["Label"].str.upper() != "BENIGN"
    ).astype(int)

    print("\nSampled records:", len(sampled))
    print("Benign records:", int((sampled["Target"] == 0).sum()))
    print("Attack records:", int((sampled["Target"] == 1).sum()))
    print("Unique exact feature groups:", sampled["_group_id"].nunique())
    print("\nLabel counts:")
    print(sampled["Label"].value_counts().to_string())

    return sampled, len(ambiguous_groups), ambiguous_rows


def main():
    sampled, ambiguous_group_count, ambiguous_row_count = (
        load_and_prepare_data()
    )

    X = sampled[FEATURES]
    y = sampled["Target"].to_numpy()
    labels = sampled["Label"].to_numpy()
    groups = sampled["_group_id"].to_numpy()

    # Split by exact feature groups. A group cannot appear in both
    # training and test data.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    try:
        train_idx, test_idx = next(
            splitter.split(X, labels, groups=groups)
        )
    except ValueError as exc:
        raise RuntimeError(
            "Could not create a stratified group split. "
            "Check whether each label has enough distinct feature groups."
        ) from exc

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    labels_test = labels[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    overlapping_groups = set(groups_train).intersection(set(groups_test))

    if overlapping_groups:
        raise RuntimeError(
            "Data leakage check failed: feature groups overlap "
            "between training and test sets."
        )

    print("\nTraining records:", len(X_train))
    print("Test records:", len(X_test))
    print("Training feature groups:", len(np.unique(groups_train)))
    print("Test feature groups:", len(np.unique(groups_test)))
    print("Overlapping feature groups:", len(overlapping_groups))

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= PROBABILITY_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    benign_mask = y_test == 0
    attack_mask = y_test == 1

    benign_fpr = (
        float(np.mean(predictions[benign_mask] == 1))
        if benign_mask.any() else 0.0
    )
    attack_recall = (
        float(np.mean(predictions[attack_mask] == 1))
        if attack_mask.any() else 0.0
    )

    try:
        auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        auc = None

    per_label = []

    for label in sorted(np.unique(labels_test)):
        mask = labels_test == label
        total = int(mask.sum())
        predicted_attack = predictions[mask]

        if label.upper() == "BENIGN":
            false_alarms = int(predicted_attack.sum())
            per_label.append({
                "label": str(label),
                "records_tested": total,
                "detected_as_attack": false_alarms,
                "missed": total - false_alarms,
                "recall_percent": None,
                "false_alarm_percent": round(
                    100 * false_alarms / total, 2
                ) if total else None,
            })
        else:
            detected = int(predicted_attack.sum())
            per_label.append({
                "label": str(label),
                "records_tested": total,
                "detected_as_attack": detected,
                "missed": total - detected,
                "recall_percent": round(
                    100 * detected / total, 2
                ) if total else None,
                "false_alarm_percent": None,
            })

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    report = {
        "experiment": "Random Forest exact-feature-group evaluation",
        "dataset": DATA_PATH.name,
        "features": FEATURES,
        "sampling": {
            "max_benign": MAX_BENIGN,
            "max_per_attack": MAX_PER_ATTACK,
            "random_state": RANDOM_STATE,
        },
        "split": {
            "method": "StratifiedGroupKFold",
            "folds": N_SPLITS,
            "evaluated_fold": 1,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "training_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "training_feature_groups": int(len(np.unique(groups_train))),
            "test_feature_groups": int(len(np.unique(groups_test))),
            "overlapping_feature_groups": int(len(overlapping_groups)),
        },
        "ambiguous_feature_groups_excluded": {
            "groups": int(ambiguous_group_count),
            "rows": int(ambiguous_row_count),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "attack_recall": float(attack_recall),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "benign_false_positive_rate": float(benign_fpr),
            "roc_auc": auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "per_label_results": per_label,
        "feature_importance": importance.to_dict(orient="records"),
        "limitations": [
            "Exact feature groups are separated between training and testing.",
            "Near duplicates may still exist.",
            "The source dataset may overlap with previous experiments.",
            "A random group split does not establish generalization to a "
            "different network, capture session, or unseen attack family.",
            "This is a supervised binary classifier, not a zero-day detector.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Group-Split Test Results ===")
    print(f"Accuracy: {100 * report['metrics']['accuracy']:.2f}%")
    print(f"Precision: {100 * report['metrics']['precision']:.2f}%")
    print(f"Attack recall: {100 * attack_recall:.2f}%")
    print(f"F1 score: {100 * report['metrics']['f1_score']:.2f}%")
    print(f"Benign false-positive rate: {100 * benign_fpr:.2f}%")
    print(f"ROC-AUC: {auc}")

    print("\nConfusion matrix:")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    print("\n=== Per-Label Detection ===")
    for row in per_label:
        if row["label"].upper() == "BENIGN":
            print(
                f"{row['label']}: false alarms="
                f"{row['false_alarm_percent']}%"
            )
        else:
            print(
                f"{row['label']}: recall={row['recall_percent']}%, "
                f"missed={row['missed']}"
            )

    print("\n=== Feature Importance ===")
    print(importance.to_string(index=False))

    print("\nCandidate model saved:", MODEL_PATH)
    print("Report saved:", REPORT_PATH)
    print("Feature importance saved:", IMPORTANCE_PATH)
    print("Production model, scaler, and threshold were not modified.")


if __name__ == "__main__":
    main()
