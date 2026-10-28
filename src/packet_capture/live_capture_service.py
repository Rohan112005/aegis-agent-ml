"""Threaded live flow capture for authorized interfaces.

The service uses the project's existing FlowAggregator and trained detector.
It does not change model weights, preprocessing, or the existing manual API.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time
from typing import Any

from src.ml_engine.flow_aggregator import FEATURE_NAMES, FlowAggregator


class LiveCaptureManager:
    def __init__(self, detector, max_events: int = 200):
        self.detector = detector
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._aggregator: FlowAggregator | None = None
        self._events = deque(maxlen=max_events)
        self._status: dict[str, Any] = {
            "running": False,
            "state": "stopped",
            "interface": None,
            "packet_filter": "ip",
            "started_at": None,
            "packets_seen": 0,
            "flows_analyzed": 0,
            "alerts": 0,
            "last_error": None,
            "message": "Live capture has not been started.",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            return list(self._events)[-limit:][::-1]

    def start(
        self,
        interface: str | None = None,
        packet_filter: str = "ip",
        idle_timeout: float = 5.0,
    ) -> dict[str, Any]:
        if not getattr(self.detector, "model_loaded", False):
            raise RuntimeError("The trained Autoencoder model is not loaded.")
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be greater than zero.")
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self._status)
            self._stop = threading.Event()
            self._aggregator = FlowAggregator(idle_timeout=idle_timeout)
            self._status = {
                **self._status,
                "running": True,
                "state": "starting",
                "interface": interface or "default",
                "packet_filter": packet_filter,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "packets_seen": 0,
                "flows_analyzed": 0,
                "alerts": 0,
                "last_error": None,
                "message": "Capture worker is starting.",
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(interface, packet_filter),
                name="aegis-live-capture",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=4.0)
        with self._lock:
            if not (thread and thread.is_alive()):
                self._status["running"] = False
                if self._status["state"] != "error":
                    self._status["state"] = "stopped"
                    self._status["message"] = "Live capture stopped."
            return dict(self._status)

    def _record_packet(self, packet) -> None:
        if self._aggregator is None:
            return
        with self._lock:
            self._status["packets_seen"] += 1
        for flow in self._aggregator.process_packet(packet):
            self._analyze_flow(flow)

    def _analyze_flow(self, flow: dict[str, Any]) -> None:
        try:
            names = flow.get("feature_names")
            features = flow.get("telemetry_features")
            if names != FEATURE_NAMES or not isinstance(features, list) or len(features) != 10:
                raise ValueError("Captured flow feature order does not match the trained model.")
            prediction = self.detector.predict(features)
            event = {
                "id": f"{time.time_ns()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "src_ip": flow.get("src_ip"),
                "dst_ip": flow.get("dst_ip"),
                "src_port": flow.get("src_port"),
                "dst_port": flow.get("dst_port"),
                "protocol": flow.get("protocol"),
                "packet_count": flow.get("packet_count"),
                "byte_count": flow.get("byte_count"),
                "anomaly": bool(prediction["anomaly"]),
                "score": float(prediction["score"]),
                "threshold": float(prediction["threshold"]),
                "status": prediction.get("status", "UNKNOWN"),
                "out_of_training_range": prediction.get("out_of_training_range", False),
                "out_of_range_features": prediction.get("out_of_range_features", []),
                "model": "autoencoder",
            }
            with self._lock:
                self._events.append(event)
                self._status["flows_analyzed"] += 1
                if event["anomaly"]:
                    self._status["alerts"] += 1
        except Exception as exc:
            with self._lock:
                self._status["last_error"] = str(exc)
                self._status["message"] = "A flow could not be analyzed; see last_error."

    def _run(self, interface: str | None, packet_filter: str) -> None:
        try:
            from scapy.all import sniff
        except ImportError:
            with self._lock:
                self._status.update(
                    running=False,
                    state="error",
                    last_error="Scapy is not installed. Run: pip install scapy",
                    message="Install Scapy before starting packet capture.",
                )
            return

        with self._lock:
            self._status["state"] = "running"
            self._status["message"] = "Capturing authorized IPv4 traffic."
        try:
            while not self._stop.is_set():
                sniff(
                    iface=interface or None,
                    prn=self._record_packet,
                    timeout=1,
                    filter=packet_filter or None,
                    store=False,
                )
                if self._aggregator is not None:
                    for flow in self._aggregator.expire(time.time()):
                        self._analyze_flow(flow)
        except Exception as exc:
            with self._lock:
                self._status.update(
                    running=False,
                    state="error",
                    last_error=str(exc),
                    message=(
                        "Capture failed. On Windows, install Npcap and run "
                        "the backend terminal with suitable permissions."
                    ),
                )
        finally:
            if self._aggregator is not None:
                for flow in self._aggregator.flush():
                    self._analyze_flow(flow)
            with self._lock:
                self._status["running"] = False
                if self._status["state"] != "error":
                    self._status["state"] = "stopped"
                    self._status["message"] = "Live capture stopped."
