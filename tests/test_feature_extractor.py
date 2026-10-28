
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw

from src.ml_engine.feature_extractor import (
    FEATURE_NAMES,
    extract_packet_features,
)


def test_tcp_packet_feature_shape():
    packet = (
        IP(src="192.168.1.10", dst="192.168.1.20")
        / TCP(sport=12345, dport=443, flags="S")
        / Raw(load=b"test payload")
    )

    features = extract_packet_features(packet)

    assert features.shape == (10,)
    assert features.dtype == "float32"


def test_udp_packet_feature_shape():
    packet = (
        IP(src="192.168.1.10", dst="192.168.1.20")
        / UDP(sport=12345, dport=53)
        / Raw(load=b"DNS payload")
    )

    features = extract_packet_features(packet)

    assert features.shape == (10,)


def test_features_are_normalized():
    packet = (
        IP()
        / TCP(sport=1234, dport=80)
        / Raw(load=b"hello")
    )

    features = extract_packet_features(packet)

    assert len(features) == len(FEATURE_NAMES)
    assert (features >= 0).all()
    assert (features <= 1).all()


def test_tcp_flag_feature():
    packet = IP() / TCP(flags="S")

    features = extract_packet_features(packet)

    assert features[8] == 1.0
    assert features[9] == 0.0
