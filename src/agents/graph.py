
import os
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentState


# Initialize Gemini only when an API key is available.
# This prevents the FastAPI backend from crashing during import.
llm = None

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"Gemini initialization failed: {exc}")
        llm = None


def invoke_gemini(prompt: str) -> str:
    """
    Call Gemini if configured.
    Return a safe fallback message if Gemini is unavailable.
    """
    if llm is None:
        return (
            "Gemini is not configured. "
            "Set GOOGLE_API_KEY to enable AI-based analysis."
        )

    try:
        response = llm.invoke(prompt)

        # Gemini responses are usually strings, but handle other content types.
        if isinstance(response.content, str):
            return response.content

        return str(response.content)

    except Exception as exc:
        print(f"Gemini request failed: {exc}")
        return "Gemini request failed. Manual review is required."


def log_investigator_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze raw telemetry and extract suspicious indicators.
    """
    telemetry = state.get("raw_telemetry", {})

    prompt = (
        f"Analyze this network telemetry: {telemetry}\n\n"
        "Extract the following information:\n"
        "1. Source and destination IP addresses\n"
        "2. Source and destination ports\n"
        "3. Suspicious protocols\n"
        "4. Suspicious payload artifacts\n"
        "5. Reasons the traffic may be anomalous\n\n"
        "Return the result as valid JSON. "
        "Do not invent information that is not present in the telemetry."
    )

    analysis = invoke_gemini(prompt)

    return {
        "parsed_logs": {
            "analysis": analysis
        },
        "status": "LOGS_PARSED",
    }


def threat_intel_node(state: AgentState) -> Dict[str, Any]:
    """
    Map suspicious behavior to MITRE ATT&CK techniques.
    """
    parsed = state.get("parsed_logs", {})

    prompt = (
        f"Analyze this network log analysis: {parsed}\n\n"
        "Identify potentially relevant MITRE ATT&CK tactics and techniques.\n"
        "Include technique IDs, names, and explanations.\n"
        "Only provide a mapping when the evidence supports it.\n"
        "If the evidence is insufficient, clearly state that.\n"
        "Return the result as valid JSON."
    )

    mitre_details = invoke_gemini(prompt)

    return {
        "mitre_mapping": {
            "mitre_details": mitre_details
        },
        "status": "THREAT_MAPPED",
    }


def remediation_planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a cautious remediation plan.
    """
    mitre = state.get("mitre_mapping", {})

    prompt = (
        f"Based on this MITRE ATT&CK analysis: {mitre}\n\n"
        "Create a structured remediation plan as valid JSON with these keys:\n"
        "'action', 'target', 'urgency', and 'reason'.\n"
        "Recommend defensive actions only.\n"
        "Do not claim that an action has already been performed.\n"
        "Do not recommend automatically blocking traffic based only on an "
        "anomaly score. Include human review when evidence is uncertain."
    )

    remediation = invoke_gemini(prompt)

    return {
        "action_plan": {
            "remediation": remediation
        },
        "status": "COMPLETED",
    }


# Build the LangGraph workflow.
workflow = StateGraph(AgentState)

workflow.add_node("investigate", log_investigator_node)
workflow.add_node("threat_intel", threat_intel_node)
workflow.add_node("remediation", remediation_planner_node)

workflow.add_edge(START, "investigate")
workflow.add_edge("investigate", "threat_intel")
workflow.add_edge("threat_intel", "remediation")
workflow.add_edge("remediation", END)

aegis_agent_graph = workflow.compile()
