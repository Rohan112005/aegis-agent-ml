
"""Aggregate packets into bidirectional CICIDS2017-compatible flow features.

Feature order must match src/ml_engine/train_cicids.py exactly.
Flow Duration is measured in microseconds.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


FEATURE_NAMES = [
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


def _endpoint(ip: str, port: int) -> tuple[str, int]:
    return (ip, int(port))


def _packet_details(packet):
    """Return flow details for supported IPv4 packets."""
    from scapy.layers.inet import IP, TCP, UDP

    if not packet.haslayer(IP):
        return None

    ip = packet[IP]
    protocol = int(ip.proto)

    if packet.haslayer(TCP):
        transport = packet[TCP]
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
        flags = int(transport.flags)

        syn = int(bool(flags & 0x02))
        ack = int(bool(flags & 0x10))

    elif packet.haslayer(UDP):
        transport = packet[UDP]
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
        syn = 0
        ack = 0

    else:
        src_port = 0
        dst_port = 0
        syn = 0
        ack = 0

    src = _endpoint(ip.src, src_port)
    dst = _endpoint(ip.dst, dst_port)

    # Use one key for both directions of the same flow.
    left, right = sorted((src, dst))
    key = (protocol, left, right)

    timestamp = float(getattr(packet, "time", 0.0))
    packet_length = float(len(ip))

    if not np.isfinite(timestamp) or timestamp < 0:
        return None

    return (
        key,
        src,
        dst,
        protocol,
        packet_length,
        syn,
        ack,
        timestamp,
    )


@dataclass
class _Flow:
    key: tuple
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float
    last_time: float

    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: float = 0.0
    bwd_bytes: float = 0.0
    syn_count: int = 0
    ack_count: int = 0
    packet_lengths: list[float] = field(default_factory=list)

    def add(self, src, dst, length, syn, ack, timestamp):
        """Add one packet to this bidirectional flow."""
        if (
            src == (self.src_ip, self.src_port)
            and dst == (self.dst_ip, self.dst_port)
        ):
            self.fwd_packets += 1
            self.fwd_bytes += length
        else:
            self.bwd_packets += 1
            self.bwd_bytes += length

        # Keep these features in the 0/1 range observed by the
        # current training scaler. This is a compatibility measure;
        # verify the exact CICIDS feature semantics separately.
        self.syn_count = max(self.syn_count, int(syn))
        self.ack_count = max(self.ack_count, int(ack))

        self.packet_lengths.append(length)
        self.last_time = max(self.last_time, timestamp)

    def to_record(self) -> dict[str, Any]:
        total_packets = self.fwd_packets + self.bwd_packets
        total_bytes = self.fwd_bytes + self.bwd_bytes

        duration_seconds = max(self.last_time - self.start_time, 0.0)
        duration_microseconds = duration_seconds * 1_000_000.0

        # Avoid division by zero. A very short flow can still produce
        # rates outside the training range, so these need validation.
        if duration_seconds > 0:
            bytes_per_second = total_bytes / duration_seconds
            packets_per_second = total_packets / duration_seconds
        else:
            bytes_per_second = 0.0
            packets_per_second = 0.0

        average_packet_size = (
            total_bytes / total_packets if total_packets else 0.0
        )

        features = np.asarray(
            [
                duration_microseconds,
                float(self.fwd_packets),
                float(self.bwd_packets),
                self.fwd_bytes,
                self.bwd_bytes,
                bytes_per_second,
                packets_per_second,
                float(self.syn_count),
                float(self.ack_count),
                average_packet_size,
            ],
            dtype=np.float32,
        )

        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "packet_count": total_packets,
            "byte_count": total_bytes,
            "raw_packet_summary": "Aggregated bidirectional flow",
            "feature_names": FEATURE_NAMES.copy(),
            "telemetry_features": features.tolist(),
        }


class FlowAggregator:
    """Build bidirectional flow records from IPv4 packets.

    A flow is emitted after it has been idle for idle_timeout seconds.
    Call flush() when capture stops to emit all remaining flows.
    """

    def __init__(self, idle_timeout: float = 5.0):
        if not np.isfinite(idle_timeout) or idle_timeout <= 0:
            raise ValueError("idle_timeout must be a positive finite number")

        self.idle_timeout = float(idle_timeout)
        self.flows: dict[tuple, _Flow] = {}

    def process_packet(self, packet) -> list[dict[str, Any]]:
        details = _packet_details(packet)

        if details is None:
            return []

        (
            key,
            src,
            dst,
            protocol,
            length,
            syn,
            ack,
            timestamp,
        ) = details

        # Emit flows that were already idle before this packet arrived.
        completed = self.expire(timestamp)

        flow = self.flows.get(key)

        if flow is None:
            flow = _Flow(
                key=key,
                src_ip=src[0],
                dst_ip=dst[0],
                src_port=src[1],
                dst_port=dst[1],
                protocol=protocol,
                start_time=timestamp,
                last_time=timestamp,
            )
            self.flows[key] = flow

        flow.add(src, dst, length, syn, ack, timestamp)

        return completed

    def expire(self, now: float) -> list[dict[str, Any]]:
        """Emit flows that have been idle for at least idle_timeout."""
        now = float(now)

        if not np.isfinite(now):
            return []

        expired_keys = [
            key
            for key, flow in self.flows.items()
            if now - flow.last_time >= self.idle_timeout
        ]

        completed = []

        for key in expired_keys:
            flow = self.flows.pop(key)
            completed.append(flow.to_record())

        return completed

    def flush(self) -> list[dict[str, Any]]:
        """Emit all remaining flows and clear the aggregator."""
        completed = [
            flow.to_record()
            for flow in self.flows.values()
        ]

        self.flows.clear()
        return completed
