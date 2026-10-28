
from scapy.layers.inet import IP, TCP, UDP

from src.ml_engine.flow_aggregator import FEATURE_NAMES, FlowAggregator


def test_flow_features_match_cicids_order_and_aggregate_directions():
    aggregator = FlowAggregator(idle_timeout=5.0)

    forward = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(
        sport=12345, dport=443, flags="SA"
    )
    forward.time = 1000.0

    backward = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(
        sport=443, dport=12345, flags="A"
    )
    backward.time = 1001.0

    assert aggregator.process_packet(forward) == []
    assert aggregator.process_packet(backward) == []

    completed = aggregator.flush()
    assert len(completed) == 1

    record = completed[0]
    features = record["telemetry_features"]

    # Check feature names and order.
    assert record["feature_names"] == FEATURE_NAMES
    assert len(features) == 10

    # Check flow duration and packet direction counts.
    assert features[0] == 1_000_000.0
    assert features[1] == 1.0
    assert features[2] == 1.0

    # SYN and ACK features stay within the training range (0 or 1).
    assert features[7] == 1.0
    assert features[8] == 1.0

    # Average packet size must be positive.
    assert features[9] > 0.0


def test_flow_expires_after_idle_timeout():
    aggregator = FlowAggregator(idle_timeout=5.0)

    packet = IP(src="10.0.0.1", dst="10.0.0.2") / UDP(
        sport=5000, dport=53
    )
    packet.time = 100.0

    aggregator.process_packet(packet)

    completed = aggregator.expire(106.0)

    assert len(completed) == 1
    assert completed[0]["telemetry_features"][1] == 1.0
    assert aggregator.flush() == []
