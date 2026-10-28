"""Unit tests for AgentState schema."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.state import AgentState


def test_agent_state_keys() -> None:
    state: AgentState = {
        "raw_telemetry": {"src_ip": "10.0.0.1"},
        "reconstruction_score": 0.12,
        "parsed_logs": {},
        "mitre_mapping": {},
        "action_plan": {},
        "status": "ANOMALY_DETECTED",
    }
    assert state["status"] == "ANOMALY_DETECTED"
    assert isinstance(state["reconstruction_score"], float)
    assert "src_ip" in state["raw_telemetry"]


def test_agent_state_accepts_empty_dicts() -> None:
    state: AgentState = {
        "raw_telemetry": {},
        "reconstruction_score": 0.0,
        "parsed_logs": {},
        "mitre_mapping": {},
        "action_plan": {},
        "status": "INIT",
    }
    assert state["reconstruction_score"] == 0.0
