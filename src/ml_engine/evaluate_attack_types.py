
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.ml_engine.autoencoder import PacketAutoencoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
REPORT_PATH = ROOT / "reports" / "attack_type_evaluation.json"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
MAX_PER_ATTACK = 10000
RANDOM_STATE = 2026


def load_threshold():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "threshold" in data:
        return float(data["threshold"])

    if "candidate_threshold" in data:
        raise ValueError(
            "Unexpected candidate threshold file. "
            "Check models/threshold.json."
        )

    raise ValueError(
        "Could not find 'threshold' in models/threshold.json. "
        f"Available keys: {list(data.keys())}"
    )


def score_flows(model, scaler, frame):
    values = frame[FEATURES].astype(np.float64)
    scaled = scaler.transform(values)
    scores = []

    with torch.no_grad():
        for start in range(0, len(scaled), BATCH_SIZE):
            batch = torch.as_tensor(
                scaled[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )

            reconstructed = model(batch)
            errors = ((reconstructed - batch) ** 2).mean(dim=1)
            scores.extend(errors.cpu().numpy().tolist())

    return np.asarray(scores)


def main():
    print("Device:", DEVICE)
    print("Loading production model and dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [c for c in FEATURES + ["Label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    benign = df[df["Label"].str.upper() == "BENIGN"]
    attacks = df[df["Label"].str.upper() != "BENIGN"]

    # Keep the evaluation manageable while retaining each attack category.
    sampled_attacks = []
    for label, group in attacks.groupby("Label"):
        sampled_attacks.append(
            group.sample(
                n=min(len(group), MAX_PER_ATTACK),
                random_state=RANDOM_STATE,
            )
        )

    if not sampled_attacks:
        raise ValueError("No attack records found in the dataset.")

    attack_test = pd.concat(sampled_attacks, ignore_index=True)

    model = PacketAutoencoder(input_dim=len(FEATURES)).to(DEVICE)

    try:
        state = torch.load(
            MODEL_PATH, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    threshold = load_threshold()

    print(f"Production threshold: {threshold:.12g}")
    print(f"Benign records available: {len(benign)}")
    print(f"Attack records sampled: {len(attack_test)}")

    print("\nScoring benign records...")
    benign_scores = score_flows(model, scaler, benign)

    benign_fp = int(np.sum(benign_scores > threshold))
    benign_fpr = (
        benign_fp / len(benign_scores) if len(benign_scores) else 0.0
    )

    print("\nScoring attack records...")
    attack_scores = score_flows(model, scaler, attack_test)

    attack_test = attack_test.copy()
    attack_test["score"] = attack_scores
    attack_test["detected"] = attack_test["score"] > threshold

    rows = []
    for label, group in attack_test.groupby("Label"):
        total = len(group)
        detected = int(group["detected"].sum())
        missed = total - detected

        rows.append({
            "attack_type": str(label),
            "records_tested": int(total),
            "detected": detected,
            "missed": missed,
            "recall_percent": round(100 * detected / total, 2),
            "average_score": float(group["score"].mean()),
            "median_score": float(group["score"].median()),
        })

    rows.sort(key=lambda item: item["recall_percent"])

    print("\n=== Attack Detection by Type (weakest first) ===")
    print(
        f"{'Attack type':32} {'Tested':>8} {'Detected':>9} "
        f"{'Missed':>8} {'Recall':>9}"
    )
    print("-" * 72)

    for row in rows:
        print(
            f"{row['attack_type'][:32]:32} "
            f"{row['records_tested']:8d} "
            f"{row['detected']:9d} "
            f"{row['missed']:8d} "
            f"{row['recall_percent']:8.2f}%"
        )

    print(f"\nBenign false-positive rate: {benign_fpr * 100:.2f}%")
    print(f"Benign false alarms: {benign_fp} / {len(benign_scores)}")

    report = {
        "model": "production",
        "threshold": threshold,
        "dataset": DATA_PATH.name,
        "device": str(DEVICE),
        "benign_records_tested": int(len(benign_scores)),
        "benign_false_positives": benign_fp,
        "benign_false_positive_rate": float(benign_fpr),
        "attack_records_tested": int(len(attack_test)),
        "attack_results_weakest_first": rows,
        "limitations": [
            "Evaluation uses the Wednesday subset of CICIDS2017.",
            "This dataset may overlap with data used during training.",
            "Results do not establish performance on unseen real-world traffic.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
