
import numpy as np
import pytest

from src.ml_engine.detector import AnomalyDetector
from src.ml_engine.feature_extractor import extract_packet_features

from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw


@pytest.fixture
def detector():
    return AnomalyDetector(
        model_path="models/autoencoder.pth",
        threshold=0.01,
    )


def test_detector_returns_expected_fields(detector):
    packet = (
        IP()
        / TCP(sport=12345, dport=443, flags="S")
        / Raw(load=b"hello")
    )

    features = extract_packet_features(packet)

    result = detector.predict(features)

    assert "anomaly" in result
    assert "score" in result
    assert "threshold" in result
    assert "status" in result


def test_detector_score_is_non_negative(detector):
    packet = IP() / UDP(sport=12345, dport=53)

    features = extract_packet_features(packet)

    result = detector.predict(features)

    assert result["score"] >= 0


def test_detector_rejects_wrong_feature_length(detector):
    wrong_features = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError):
        detector.predict(wrong_features)


def test_detector_rejects_nan_features(detector):
    invalid_features = np.array(
        [0.1] * 9 + [np.nan],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        detector.predict(invalid_features)
