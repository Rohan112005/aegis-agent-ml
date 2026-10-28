from typing import TypedDict, Dict, Any


class AgentState(TypedDict):
    raw_telemetry: Dict[str, Any]
    reconstruction_score: float
    parsed_logs: Dict[str, Any]
    mitre_mapping: Dict[str, Any]
    action_plan: Dict[str, Any]
    status: str
