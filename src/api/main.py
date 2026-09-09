"""
FastAPI Production Inference Microservice for B2B SaaS Churn Early-Warning.

Provides real-time scoring, SHAP local explainability, batch inference,
interactive dashboard, and Kubernetes-compatible health/readiness probes.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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
state: Dict[str, Any] = {}

# Template directory for dashboard
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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


# ---------------------------------------------------------------------------
# Dashboard Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
async def dashboard(request: Request):
    """Serve the interactive Account Health Command Center dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html", context={})


@app.get("/api/model-metadata", tags=["Dashboard"], summary="Model performance metrics")
async def model_metadata():
    """Return model training/test metrics and metadata for dashboard display."""
    meta = state.get("metadata", {})
    return {
        "model_version": meta.get("model_version", "1.0.0"),
        "model_type": meta.get("model_type", "LightGBMClassifier"),
        "optimal_threshold": meta.get("optimal_threshold", 0.5),
        "test_metrics": meta.get("test_metrics", {}),
        "train_metrics": meta.get("train_metrics", {}),
        "baselines": meta.get("baselines", {}),
        "churn_rate_test": meta.get("churn_rate_test", 0.0),
    }


@app.get(
    "/api/sample-accounts",
    response_model=BatchPredictionResponse,
    tags=["Dashboard"],
    summary="Generate and score sample accounts for dashboard display",
)
async def sample_accounts():
    """
    Generate 20 diverse synthetic accounts spanning all risk tiers,
    score them through the full pipeline, and return batch results
    for the Account Health Command Center table.
    """
    rng = np.random.default_rng(seed=int(time.time()) % 10000)
    accounts_raw: List[Dict[str, Any]] = []

    tier_configs = {
        "Enterprise": {"mrr_range": (15000, 85000), "seats_range": (150, 800)},
        "Mid-Market": {"mrr_range": (5000, 25000), "seats_range": (30, 200)},
        "SMB": {"mrr_range": (800, 6000), "seats_range": (5, 40)},
    }

    tiers = rng.choice(["Enterprise", "Mid-Market", "SMB"], size=20, p=[0.25, 0.45, 0.30])

    for i, tier in enumerate(tiers):
        cfg = tier_configs[tier]
        mrr = round(float(rng.uniform(*cfg["mrr_range"])), 2)
        seats = int(rng.integers(*cfg["seats_range"]))
        seat_util = round(float(rng.uniform(0.15, 0.95)), 2)
        active_users = max(1, int(seats * seat_util))
        login_decay = round(float(rng.uniform(0.20, 1.0)), 2)
        login_freq = round(float(rng.uniform(1.0, 20.0)), 1)
        feature_adopt = round(float(rng.uniform(0.10, 0.95)), 2)
        tickets_90d = int(rng.integers(0, 30))
        p1_tickets = int(rng.integers(0, 8))
        escalation_vel = round(float(rng.uniform(0.0, 5.0)), 2)
        csm_touch = int(rng.integers(0, 10))
        billing_overdue = int(rng.choice([0, 0, 0, 5, 10, 18, 25, 35, 50]))
        contract_months = int(rng.choice([12, 12, 24, 36]))
        tenure = int(rng.integers(2, 48))
        nps = round(float(rng.uniform(-50, 80)), 0) if rng.random() > 0.15 else None

        accounts_raw.append(
            {
                "account_id": f"ACC-{10000 + i}",
                "contract_tier": tier,
                "contract_duration_months": contract_months,
                "tenure_months": tenure,
                "monthly_recurring_revenue": mrr,
                "licensed_seats": seats,
                "active_users_last_30d": active_users,
                "seat_utilization_ratio": seat_util,
                "login_frequency_last_30d": login_freq,
                "login_decay_ratio": login_decay,
                "feature_adoption_score": feature_adopt,
                "support_tickets_last_90d": tickets_90d,
                "p1_tickets_last_30d": p1_tickets,
                "ticket_escalation_velocity": escalation_vel,
                "csm_touchpoints_last_90d": csm_touch,
                "billing_overdue_days": billing_overdue,
                "nps_score": nps,
            }
        )

    # Score all accounts through the pipeline
    t0 = time.perf_counter()
    predictions: List[PredictionResponse] = []

    for raw in accounts_raw:
        payload = AccountTelemetryInput(**raw)
        pred = _score_single_account(payload)
        predictions.append(pred)

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


# ---------------------------------------------------------------------------
# System Endpoints
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------


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
        contract_tier=payload.contract_tier,
        monthly_recurring_revenue=payload.monthly_recurring_revenue,
    )


# ---------------------------------------------------------------------------
# Inference Endpoints
# ---------------------------------------------------------------------------


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
