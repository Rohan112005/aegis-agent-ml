"""FastAPI endpoints for AegisAgent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ml_engine.detector import AnomalyDetector
from src.ml_engine.feature_extractor import FEATURE_NAMES, extract_features
import src.agents.graph as graph_module


app = FastAPI(
    title="AegisAgent API",
    description="Network anomaly detection and investigation",
    version="0.2.0",
)
detector = AnomalyDetector()


class AnalyzeRequest(BaseModel):
    telemetry_features: list[float] = Field(..., min_length=10, max_length=10)
    raw_log: dict = Field(default_factory=dict)


class ExtractRequest(BaseModel):
    packet: dict = Field(default_factory=dict)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.model_loaded,
        "threshold": detector.threshold,
    }


@app.get("/api/dashboard")
async def dashboard():
    return {
        "status": "ok",
        "model_loaded": detector.model_loaded,
        "threshold": detector.threshold,
        "input_dim": 10,
        "capture_status": "not_started",
        "message": "Dashboard API ready. Live capture requires authorized local setup.",
    }


@app.post("/extract-features")
async def extract(request: ExtractRequest):
    return {
        "feature_names": FEATURE_NAMES,
        "feature_type": "normalized_packet_features_not_CICIDS_flow_features",
        "telemetry_features": extract_features(request.packet).tolist(),
    }


@app.post("/analyze-packet")
async def analyze_packet(request: AnalyzeRequest):
    try:
        result = detector.predict(request.telemetry_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if result["anomaly"]:
        state = {
            "raw_telemetry": request.raw_log,
            "reconstruction_score": result["score"],
            "parsed_logs": {},
            "mitre_mapping": {},
            "action_plan": {},
            "status": "ANOMALY_DETECTED",
        }

        try:
            final_state = graph_module.aegis_agent_graph.invoke(state)
        except Exception as exc:
            final_state = {
                "status": "AGENT_UNAVAILABLE",
                "detail": str(exc),
            }

        result["agent_response"] = final_state
        result["status"] = "ANOMALY_DETECTED"
    else:
        result["status"] = "TRAFFIC_NORMAL"

    return result
