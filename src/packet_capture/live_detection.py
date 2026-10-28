
"""Live AegisAgent flow detection with CSV feature logging."""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.ml_engine.detector import AnomalyDetector
from src.ml_engine.rf_detector import RandomForestDetector
from src.ml_engine.flow_aggregator import FEATURE_NAMES
from src.packet_capture.live_monitor import capture_packets


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "reports" / "live_flow_features.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "timestamp_utc",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "packet_count",
    "byte_count",
    *FEATURE_NAMES,
    "anomaly",
    "score",
    "threshold",
    "status",
    "model",
]


def detect_flow(record: dict, detector) -> dict:
    """Validate a completed flow and classify it with the selected model."""

    record_features = record.get("feature_names")
    features = record.get("telemetry_features")

    if record_features != FEATURE_NAMES:
        raise ValueError(
            "Flow feature names or order do not match the trained model."
        )

    if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} flow features.")

    prediction = detector.predict(features)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "src_ip": record.get("src_ip"),
        "dst_ip": record.get("dst_ip"),
        "src_port": record.get("src_port"),
        "dst_port": record.get("dst_port"),
        "protocol": record.get("protocol"),
        "packet_count": record.get("packet_count"),
        "byte_count": record.get("byte_count"),
        "feature_names": FEATURE_NAMES.copy(),
    }

    for name, value in zip(FEATURE_NAMES, features):
        result[name] = float(value)

    result.update(
        {
            "anomaly": bool(prediction["anomaly"]),
            "score": float(prediction["score"]),
            "threshold": float(prediction["threshold"]),
            "status": prediction["status"],
            "model": prediction.get("model", "autoencoder"),
        }
    )

    return result


def append_csv_record(csv_path: Path, result: dict) -> None:
    """Append one flow record to CSV, creating the header if necessary."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = (
        not csv_path.exists() or csv_path.stat().st_size == 0
    )

    csv_record = {
        key: value
        for key, value in result.items()
        if key in CSV_COLUMNS
    }

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        writer.writerow(csv_record)


def run_live_detection(
    interface=None,
    count=0,
    packet_filter="ip",
    idle_timeout=5.0,
    detector=None,
    on_result: Callable[[dict], None] | None = None,
    csv_path=DEFAULT_CSV,
    model="autoencoder",
):
    """Capture authorized traffic, classify flows, and save CSV records."""

    if count < 0:
        raise ValueError("count cannot be negative.")

    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive.")

    if model not in ("autoencoder", "random_forest"):
        raise ValueError(
            "model must be 'autoencoder' or 'random_forest'."
        )

    # An explicitly supplied detector takes precedence for tests/custom use.
    if detector is not None:
        active_detector = detector
    elif model == "random_forest":
        active_detector = RandomForestDetector()
    else:
        active_detector = AnomalyDetector()

    if not getattr(active_detector, "model_loaded", False):
        raise RuntimeError(
            "The trained model or scaler could not be loaded. "
            "Check the model artifacts in the models directory."
        )

    output_path = Path(csv_path)
    results = []

    def handle_flow(record):
        try:
            result = detect_flow(record, active_detector)
            append_csv_record(output_path, result)

        except (
            ValueError,
            RuntimeError,
            OSError,
            TypeError,
            KeyError,
        ) as exc:
            logger.error("Could not process flow: %s", exc)
            return

        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

        if on_result is not None:
            on_result(result)

    logger.info("Starting authorized live flow detection.")
    logger.info("Interface: %s", interface or "default")
    logger.info("Packet filter: %s", packet_filter)
    logger.info("Packet count limit: %s", count or "unlimited")
    logger.info("Idle timeout: %s seconds", idle_timeout)
    logger.info("Selected model: %s", model)
    logger.info("Active model threshold: %.10g", active_detector.threshold)
    logger.info("CSV output: %s", output_path)

    capture_packets(
        callback=handle_flow,
        interface=interface,
        count=count,
        filter=packet_filter,
        idle_timeout=idle_timeout,
    )

    logger.info("Capture ended. Classified flows: %d", len(results))
    logger.info("Flow records saved to: %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="AegisAgent live flow detection with CSV logging"
    )

    parser.add_argument(
        "--interface",
        default=None,
        help="Authorized capture interface; default is Scapy's default.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of packets to capture; 0 means until interrupted.",
    )

    parser.add_argument(
        "--filter",
        dest="packet_filter",
        default="ip",
        help="BPF capture filter. Default: ip",
    )

    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=5.0,
        help="Seconds of inactivity before a flow is completed.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV output path for live flow features and scores.",
    )

    parser.add_argument(
        "--model",
        choices=["autoencoder", "random_forest"],
        default="autoencoder",
        help="Detection model. Default: autoencoder.",
    )

    args = parser.parse_args()

    try:
        run_live_detection(
            interface=args.interface,
            count=args.count,
            packet_filter=args.packet_filter,
            idle_timeout=args.idle_timeout,
            csv_path=args.output,
            model=args.model,
        )

    except PermissionError as exc:
        logger.error(
            "Capture permission denied. Use an authorized interface "
            "and the required system permissions."
        )
        raise SystemExit(1) from exc

    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Live detection failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
