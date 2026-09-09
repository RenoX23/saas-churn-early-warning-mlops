"""
FastAPI Production Inference Microservice for B2B SaaS Churn Early-Warning.

Provides real-time scoring, SHAP local explainability, batch inference,
and Kubernetes-compatible health/readiness probes.
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from src.api.schemas import (
    AccountTelemetryInput,
    BatchPredictionResponse,
    BatchTelemetryInput,
    HealthResponse,
    PredictionResponse,
    RiskDriverSchema,
)
from src.explainability.explainer import SaaSChurnExplainer
from src.features.build_features import SaaSFeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# State container for serving
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to pre-load ML artifacts and initialize
    the SHAP explainer during startup for sub-35ms inference.
    """
    artifacts_dir = Path("models")
    model_path = artifacts_dir / "lgbm_model.joblib"
    transformer_path = artifacts_dir / "feature_transformer.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    if not model_path.exists() or not transformer_path.exists() or not metadata_path.exists():
        logger.error("Required model artifacts not found in %s", artifacts_dir)
        raise RuntimeError(f"Missing required model artifacts in {artifacts_dir}")

    logger.info("Loading model artifacts from %s...", artifacts_dir)
    state["model"] = joblib.load(model_path)
    state["transformer"] = SaaSFeatureTransformer.load(transformer_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        state["metadata"] = json.load(f)

    state["explainer"] = SaaSChurnExplainer(
        model=state["model"],
        feature_names=state["transformer"].feature_names_,
    )
    state["start_time"] = time.time()
    logger.info("Serving layer ready. Model version: %s", state["metadata"].get("model_version"))

    yield

    logger.info("Shutting down inference microservice...")
    state.clear()


app = FastAPI(
    title="B2B SaaS Churn Early-Warning Microservice",
    description=(
        "Production MLOps microservice providing real-time account risk scoring, "
        "calibrated probability thresholds, and SHAP top-3 actionable drivers for Customer Success."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external CRM/dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness and readiness probe",
)
async def health_check():
    """Liveness probe verifying model availability and system uptime."""
    if "model" not in state or "transformer" not in state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not yet loaded.",
        )

    uptime = round(time.time() - state.get("start_time", time.time()), 2)
    meta = state.get("metadata", {})

    return HealthResponse(
        status="healthy",
        model_version=meta.get("model_version", "1.0.0"),
        model_type=meta.get("model_type", "LightGBMClassifier"),
        optimal_threshold=float(meta.get("optimal_threshold", 0.5)),
        features_count=len(state["transformer"].feature_names_),
        uptime_seconds=uptime,
    )


def _score_single_account(payload: AccountTelemetryInput) -> PredictionResponse:
    """Internal helper to score a single account and extract SHAP risk drivers."""
    t0 = time.perf_counter()

    raw_dict = payload.model_dump()
    df_raw = pd.DataFrame([raw_dict])

    # Transform through zero-leakage feature pipeline
    transformed = state["transformer"].transform(df_raw)

    # Model inference
    prob = float(state["model"].predict_proba(transformed)[0, 1])
    optimal_th = float(state["metadata"].get("optimal_threshold", 0.5))
    predicted_churn = prob >= optimal_th

    # Determine risk tier
    if prob >= 0.70:
        risk_level = "CRITICAL"
    elif prob >= optimal_th:
        risk_level = "ELEVATED"
    else:
        risk_level = "NOMINAL"

    # SHAP local attribution
    risk_drivers_data = state["explainer"].explain_instance(
        transformed_features=transformed,
        raw_inputs=raw_dict,
        top_k=3,
    )
    risk_drivers = [RiskDriverSchema(**d) for d in risk_drivers_data]

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return PredictionResponse(
        account_id=payload.account_id,
        churn_probability=round(prob, 4),
        predicted_churn=predicted_churn,
        risk_level=risk_level,
        optimal_threshold=round(optimal_th, 4),
        top_risk_drivers=risk_drivers,
        inference_latency_ms=latency_ms,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Real-time single account churn scoring with SHAP explainability",
)
async def predict_single(payload: AccountTelemetryInput):
    """
    Score an individual B2B SaaS account:
    - Calculates calibrated churn probability
    - Assigns risk tier (CRITICAL, ELEVATED, NOMINAL)
    - Dynamically extracts the top 3 actionable SHAP risk drivers with recommended playbooks
    - Returns sub-35ms response
    """
    try:
        return _score_single_account(payload)
    except Exception as e:
        logger.exception("Error scoring account %s: %s", payload.account_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch scoring for CRM portfolio accounts",
)
async def predict_batch(payload: BatchTelemetryInput):
    """
    Score a batch of enterprise accounts in a single API call.
    Returns individual risk evaluations and summary portfolio metrics.
    """
    t0 = time.perf_counter()
    predictions: List[PredictionResponse] = []

    try:
        for account in payload.accounts:
            predictions.append(_score_single_account(account))

        total = len(predictions)
        at_risk = sum(1 for p in predictions if p.predicted_churn)
        mean_prob = round(sum(p.churn_probability for p in predictions) / max(total, 1), 4)
        total_latency = round((time.perf_counter() - t0) * 1000.0, 2)

        return BatchPredictionResponse(
            total_accounts=total,
            at_risk_count=at_risk,
            mean_churn_probability=mean_prob,
            total_latency_ms=total_latency,
            predictions=predictions,
        )
    except Exception as e:
        logger.exception("Batch prediction failure: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )
