
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "candidate_random_forest.joblib"
REPORT_PATH = ROOT / "reports" / "random_forest_baseline.json"
IMPORTANCE_PATH = ROOT / "reports" / "random_forest_feature_importance.csv"

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
TEST_SIZE = 0.25
RANDOM_STATE = 42


# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

def load_dataset():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    # Sample benign traffic.
    benign = df[df["Label"].str.upper() == "BENIGN"]
    benign = benign.sample(
        n=min(len(benign), MAX_BENIGN),
        random_state=RANDOM_STATE,
    )

    # Sample each attack category separately.
    attack_groups = []
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    for label, group in attacks.groupby("Label"):
        attack_groups.append(
            group.sample(
                n=min(len(group), MAX_PER_ATTACK),
                random_state=RANDOM_STATE,
            )
        )

    if not attack_groups:
        raise ValueError("No attack records found.")

    sampled = pd.concat(
        [benign] + attack_groups,
        ignore_index=True,
    )

    # Normalize label spelling for binary target creation.
    sampled["Label"] = sampled["Label"].str.strip()
    sampled["Target"] = (
        sampled["Label"].str.upper() != "BENIGN"
    ).astype(int)

    print("Records:", len(sampled))
    print("Benign records:", int((sampled["Target"] == 0).sum()))
    print("Attack records:", int((sampled["Target"] == 1).sum()))
    print("\nLabel distribution:")
    print(sampled["Label"].value_counts().to_string())

    return sampled


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    sampled = load_dataset()

    X = sampled[FEATURES]
    y = sampled["Target"]
    labels = sampled["Label"]

    # Split by original traffic label so each available category
    # is represented in both train and test sets.
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

    print("\nTraining records:", len(X_train))
    print("Test records:", len(X_test))

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

    # Predict using the default 0.50 probability threshold.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    benign_mask = y_test.to_numpy() == 0
    attack_mask = y_test.to_numpy() == 1

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

    # Evaluate each attack label separately.
    per_label = []

    labels_test_array = labels_test.to_numpy()

    for label in sorted(labels_test.unique()):
        mask = labels_test_array == label
        count = int(mask.sum())

        if label.upper() == "BENIGN":
            false_alarms = int(predictions[mask].sum())
            per_label.append({
                "label": label,
                "records_tested": count,
                "detected_as_attack": false_alarms,
                "missed": count - false_alarms,
                "recall_percent": None,
                "false_alarm_percent": round(
                    100 * false_alarms / count, 2
                ) if count else 0.0,
            })
        else:
            detected = int(predictions[mask].sum())
            per_label.append({
                "label": label,
                "records_tested": count,
                "detected_as_attack": detected,
                "missed": count - detected,
                "recall_percent": round(
                    100 * detected / count, 2
                ) if count else 0.0,
                "false_alarm_percent": None,
            })

    # Feature importance is a useful starting point, not proof
    # that a feature causes or uniquely identifies an attack.
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save only to a separate candidate artifact.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    report = {
        "experiment": "Random Forest binary classification baseline",
        "dataset": DATA_PATH.name,
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "probability_threshold": 0.50,
        "records": {
            "total": int(len(sampled)),
            "training": int(len(X_train)),
            "test": int(len(X_test)),
            "benign_test": int(benign_mask.sum()),
            "attack_test": int(attack_mask.sum()),
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
            "This is a supervised binary classifier, not an autoencoder.",
            "A random split from the same dataset is not independent "
            "real-world validation.",
            "Results depend on the sampled class distribution.",
            "The default probability threshold is 0.50; it is not "
            "guaranteed to produce a 5% false-positive rate.",
            "The candidate model does not replace the production model.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Overall Test Results ===")
    print(f"Accuracy:             {100 * report['metrics']['accuracy']:.2f}%")
    print(f"Precision:            {100 * report['metrics']['precision']:.2f}%")
    print(f"Attack recall:        {100 * attack_recall:.2f}%")
    print(f"F1 score:             {100 * report['metrics']['f1_score']:.2f}%")
    print(f"Benign false alarms:  {100 * benign_fpr:.2f}%")
    print(f"ROC-AUC:              {auc}")

    print("\nConfusion matrix:")
    print("True benign, flagged benign:", tn)
    print("Benign incorrectly flagged:", fp)
    print("Attacks missed:", fn)
    print("Attacks detected:", tp)

    print("\n=== Results by Label ===")
    print(classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    print("\n=== Per-Label Detection ===")
    for row in per_label:
        if row["label"].upper() == "BENIGN":
            print(
                f"{row['label']}: "
                f"false alarms={row['false_alarm_percent']}%"
            )
        else:
            print(
                f"{row['label']}: "
                f"recall={row['recall_percent']}%, "
                f"missed={row['missed']}"
            )

    print("\n=== Top Feature Importances ===")
    print(importance.head(10).to_string(index=False))

    print("\nCandidate model saved:", MODEL_PATH)
    print("Evaluation report saved:", REPORT_PATH)
    print("Feature importances saved:", IMPORTANCE_PATH)
    print("\nProduction model, scaler, and threshold were not modified.")


if __name__ == "__main__":
    main()
