
"""Tests for AegisAgent live-flow detection."""

import pytest

from src.ml_engine.flow_aggregator import FEATURE_NAMES
from src.packet_capture import live_detection


class FakeDetector:
    model_loaded = True
    threshold = 0.001

    def __init__(self):
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return {
            "anomaly": True,
            "score": 0.002,
            "threshold": self.threshold,
            "status": "ANOMALY",
            "model_loaded": True,
        }


def make_flow():
    return {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 12345,
        "dst_port": 443,
        "protocol": 6,
        "packet_count": 2,
        "byte_count": 120,
        "feature_names": FEATURE_NAMES.copy(),
        "telemetry_features": [
            1_000_000.0,
            1.0,
            1.0,
            60.0,
            60.0,
            120.0,
            2.0,
            1.0,
            2.0,
            60.0,
        ],
    }


def test_detect_flow_passes_flow_features_to_detector():
    detector = FakeDetector()
    record = make_flow()

    result = live_detection.detect_flow(record, detector)

    assert detector.received_features == record["telemetry_features"]
    assert result["anomaly"] is True
    assert result["status"] == "ANOMALY"
    assert result["score"] == 0.002
    assert result["threshold"] == 0.001
    assert result["feature_names"] == FEATURE_NAMES


def test_detect_flow_rejects_wrong_feature_order():
    detector = FakeDetector()
    record = make_flow()
    record["feature_names"] = list(reversed(FEATURE_NAMES))

    with pytest.raises(ValueError, match="feature names or order"):
        live_detection.detect_flow(record, detector)


def test_detect_flow_rejects_wrong_feature_count():
    detector = FakeDetector()
    record = make_flow()
    record["telemetry_features"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="Expected 10 flow features"):
        live_detection.detect_flow(record, detector)


def test_run_live_detection_rejects_missing_model(monkeypatch):
    class UnloadedDetector:
        model_loaded = False
        threshold = 0.001

    def should_not_capture(**kwargs):
        pytest.fail("Capture should not start without a loaded model.")

    monkeypatch.setattr(
        live_detection,
        "capture_packets",
        should_not_capture,
    )

    with pytest.raises(RuntimeError, match="model or scaler"):
        live_detection.run_live_detection(detector=UnloadedDetector())
