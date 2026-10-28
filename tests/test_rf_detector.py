
import pytest

from src.ml_engine.rf_detector import RandomForestDetector, FEATURES


def test_detector_loads_candidate_model():
    detector = RandomForestDetector()
    assert detector.model is not None


def test_detector_returns_expected_fields():
    detector = RandomForestDetector()

    example = {
        "Flow Duration": 1000000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Total Length of Fwd Packets": 5000,
        "Total Length of Bwd Packets": 2500,
        "Flow Bytes/s": 7500,
        "Flow Packets/s": 15,
        "SYN Flag Count": 1,
        "ACK Flag Count": 1,
        "Average Packet Size": 500,
    }

    result = detector.predict(example)

    assert set([
        "anomaly",
        "score",
        "threshold",
        "status",
        "model",
        "model_loaded",
    ]).issubset(result)

    assert isinstance(result["anomaly"], bool)
    assert 0.0 <= result["score"] <= 1.0
    assert result["model_loaded"] is True


def test_detector_rejects_wrong_feature_count():
    detector = RandomForestDetector()

    with pytest.raises(ValueError):
        detector.predict([1, 2, 3])


def test_detector_rejects_non_finite_features():
    detector = RandomForestDetector()
    values = [1.0] * len(FEATURES)
    values[0] = float("nan")

    with pytest.raises(ValueError):
        detector.predict(values)
