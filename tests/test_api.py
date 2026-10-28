
"""Unit tests for FastAPI endpoints (no real Gemini calls)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Create a FastAPI test client with a mocked agent graph.

    The mocked graph prevents tests from making real Gemini API calls.
    """
    mock_graph = MagicMock()

    mock_graph.invoke.return_value = {
        "raw_telemetry": {"src_ip": "1.2.3.4"},
        "reconstruction_score": 0.2,
        "parsed_logs": {"analysis": "mock"},
        "mitre_mapping": {"mitre_details": "T1059"},
        "action_plan": {
            "remediation": (
                '{"action": "isolate", '
                '"target": "host", '
                '"urgency": "high"}'
            )
        },
        "status": "COMPLETED",
    }

    # Patch the graph before importing the FastAPI application.
    with patch("src.agents.graph.aegis_agent_graph", mock_graph):
        from src.api.main import app

        yield TestClient(app), mock_graph


def test_health(client) -> None:
    """Check that the health endpoint returns a successful response."""
    c, _ = client

    response = c.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_analyze_normal_traffic(client) -> None:
    """
    Test the packet analysis endpoint.

    If the traffic is classified as an anomaly, the agent graph
    should be called. Otherwise, the response should indicate
    that the traffic is normal.
    """
    c, mock_graph = client

    payload = {
        "telemetry_features": [0.5] * 10,
        "raw_log": {"src_ip": "10.0.0.1"},
    }

    response = c.post("/analyze-packet", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "anomaly" in body
    assert "score" in body
    assert "threshold" in body

    if body["anomaly"]:
        mock_graph.invoke.assert_called_once()
        assert "agent_response" in body
    else:
        assert body.get("status") == "TRAFFIC_NORMAL"


def test_analyze_rejects_wrong_feature_length(client) -> None:
    """Reject telemetry with an incorrect number of features."""
    c, _ = client

    response = c.post(
        "/analyze-packet",
        json={
            "telemetry_features": [0.1, 0.2],
            "raw_log": {},
        },
    )

    assert response.status_code == 422
