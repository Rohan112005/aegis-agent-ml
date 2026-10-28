
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
REPORT_PATH = ROOT / "reports" / "threshold_tradeoff.json"

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
MAX_PER_LABEL = 10000
SEED = 2026


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    df = df[FEATURES + ["Label"]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    benign = df[df["Label"].str.upper() == "BENIGN"].copy()
    attacks = df[df["Label"].str.upper() != "BENIGN"].copy()

    # Reuse the same sampled records at every threshold.
    benign = benign.sample(frac=1, random_state=SEED)
    attack_parts = []
    for label, group in attacks.groupby("Label"):
        attack_parts.append(
            group.sample(
                n=min(len(group), MAX_PER_LABEL),
                random_state=SEED,
            )
        )
    attacks = pd.concat(attack_parts, ignore_index=True)

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
    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_data = json.load(f)
    base_threshold = float(threshold_data["threshold"])

    def scores(frame):
        scaled = scaler.transform(frame[FEATURES].astype(float))
        result = []
        with torch.no_grad():
            for start in range(0, len(scaled), BATCH_SIZE):
                batch = torch.as_tensor(
                    scaled[start:start + BATCH_SIZE],
                    dtype=torch.float32,
                    device=DEVICE,
                )
                reconstructed = model(batch)
                error = ((reconstructed - batch) ** 2).mean(dim=1)
                result.extend(error.cpu().numpy().tolist())
        return np.asarray(result)

    print("Scoring benign records...")
    benign_scores = scores(benign)
    print("Scoring attack records...")
    attacks["score"] = scores(attacks)

    factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    report = []

    for factor in factors:
        threshold = base_threshold * factor
        benign_fp = int(np.sum(benign_scores > threshold))
        fpr = benign_fp / len(benign_scores)

        result = {
            "threshold_factor": factor,
            "threshold": float(threshold),
            "benign_false_positives": benign_fp,
            "benign_false_positive_rate_percent": round(fpr * 100, 2),
            "attack_types": {},
        }

        for label, group in attacks.groupby("Label"):
            detected = int((group["score"] > threshold).sum())
            total = len(group)
            result["attack_types"][label] = {
                "tested": int(total),
                "detected": detected,
                "missed": total - detected,
                "recall_percent": round(100 * detected / total, 2),
            }

        report.append(result)

    print("\n=== Threshold trade-off ===")
    print(
        f"{'Factor':>7} {'Threshold':>14} {'Benign FPR':>12} "
        f"{'GoldenEye':>12} {'Slowhttptest':>14} {'Hulk':>9}"
    )
    print("-" * 76)

    for result in report:
        types = result["attack_types"]
        golden = types.get("DoS GoldenEye", {}).get("recall_percent", 0)
        slow = types.get("DoS Slowhttptest", {}).get("recall_percent", 0)
        hulk = types.get("DoS Hulk", {}).get("recall_percent", 0)

        print(
            f"{result['threshold_factor']:7.2f} "
            f"{result['threshold']:14.8g} "
            f"{result['benign_false_positive_rate_percent']:11.2f}% "
            f"{golden:11.2f}% {slow:13.2f}% {hulk:8.2f}%"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "base_threshold": base_threshold,
        "records": {
            "benign": len(benign),
            "attacks": len(attacks),
        },
        "results": report,
        "note": (
            "Exploratory analysis on CICIDS2017 Wednesday data; "
            "not independent real-world validation."
        ),
    }, indent=2), encoding="utf-8")

    print("\nSaved:", REPORT_PATH)
    print("Production model and threshold files were not modified.")


if __name__ == "__main__":
    main()
