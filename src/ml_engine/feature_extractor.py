"""Extract normalized packet features for AegisAgent."""

import numpy as np
from scapy.layers.inet import IP, TCP, UDP


FEATURE_NAMES = [
    "duration",
    "protocol",
    "src_port",
    "dst_port",
    "packet_count",
    "byte_count",
    "packet_length",
    "flow_rate",
    "syn_flag",
    "bytes_per_second",
]


def _number(record, *keys, default=0.0):
    """Return the first available numeric value from a record."""
    for key in keys:
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
    return float(default)


def _normalize(value, maximum):
    """Clamp and normalize a value to the range [0, 1]."""
    value = max(float(value), 0.0)
    if maximum <= 0:
        return 0.0
    return float(np.clip(value / maximum, 0.0, 1.0))


def extract_features(record):
    """Convert a packet/flow record to a normalized float32 vector."""

    duration = max(_number(record, "duration", "flow_duration"), 0.0)
    packets = max(
        _number(record, "packet_count", "total_packets", "packets"),
        0.0,
    )
    byte_count = max(
        _number(record, "byte_count", "total_bytes", "bytes"),
        0.0,
    )
    packet_length = max(
        _number(
            record,
            "packet_length",
            "length",
            "len",
            default=byte_count / packets if packets else 0.0,
        ),
        0.0,
    )

    flow_rate = packets / duration if duration > 0 else 0.0
    bytes_per_second = byte_count / duration if duration > 0 else 0.0

    features = [
        _normalize(duration, 60.0),
        _normalize(_number(record, "protocol", "protocol_number"), 255.0),
        _normalize(_number(record, "src_port", "sport"), 65535.0),
        _normalize(_number(record, "dst_port", "dport"), 65535.0),
        _normalize(packets, 1000.0),
        _normalize(byte_count, 1500000.0),
        _normalize(packet_length, 1500.0),
        _normalize(flow_rate, 1000.0),
        _normalize(_number(record, "syn_flag", "SYN", "tcp_syn"), 1.0),
        _normalize(bytes_per_second, 1000000.0),
    ]

    return np.asarray(features, dtype=np.float32)


def extract_packet_features(packet):
    """Extract normalized features from a Scapy packet."""

    packet_length = float(len(packet))
    record = {
        "duration": 0.0,
        "protocol": 0,
        "src_port": 0,
        "dst_port": 0,
        "packet_count": 1,
        "byte_count": packet_length,
        "packet_length": packet_length,
        "flow_rate": 0.0,
        "bytes_per_second": 0.0,
        "syn_flag": 0,
    }

    if packet.haslayer(IP):
        record["protocol"] = int(packet[IP].proto)

    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        record["src_port"] = int(tcp_layer.sport)
        record["dst_port"] = int(tcp_layer.dport)
        record["syn_flag"] = int(bool(int(tcp_layer.flags) & 0x02))

    elif packet.haslayer(UDP):
        udp_layer = packet[UDP]
        record["src_port"] = int(udp_layer.sport)
        record["dst_port"] = int(udp_layer.dport)

    return extract_features(record)
