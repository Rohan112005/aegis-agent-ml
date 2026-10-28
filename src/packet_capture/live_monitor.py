"""Capture authorized local traffic and emit completed CICIDS-style flows."""

from src.ml_engine.flow_aggregator import FlowAggregator


def capture_packets(callback, interface=None, count=0, filter="ip", idle_timeout=5.0):
    """Capture packets and call callback(record) when a flow completes.

    `count=0` captures until interrupted. Flow features are raw CICIDS-style
    statistics in the exact order used by train_cicids.py; the detector applies
    the saved StandardScaler.
    """
    try:
        from scapy.all import sniff
    except ImportError as exc:
        raise RuntimeError(
            "Scapy is required. Install it with: pip install scapy"
        ) from exc

    aggregator = FlowAggregator(idle_timeout=idle_timeout)
    packets_seen = 0

    def handle(packet):
        nonlocal packets_seen
        packets_seen += 1

        for record in aggregator.process_packet(packet):
            callback(record)

    try:
        # Short sniff windows allow idle flows to expire even when no new
        # packets arrive. Ctrl+C stops an unbounded capture.
        while count == 0 or packets_seen < count:
            remaining = 0 if count == 0 else count - packets_seen
            sniff(
                iface=interface,
                prn=handle,
                count=remaining,
                timeout=1,
                filter=filter,
                store=False,
            )
            import time
            for record in aggregator.expire(time.time()):
                callback(record)
    except KeyboardInterrupt:
        pass
    finally:
        for record in aggregator.flush():
            callback(record)
