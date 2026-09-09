"""
Unit tests for SHAP TreeExplainer and risk driver extraction.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.explainability.explainer import SaaSChurnExplainer
from src.features.build_features import SaaSFeatureTransformer


@pytest.fixture(scope="module")
def artifacts():
    """Load pre-trained artifacts for testing."""
    model_path = Path("models/lgbm_model.joblib")
    trans_path = Path("models/feature_transformer.joblib")
    if not model_path.exists() or not trans_path.exists():
        pytest.skip("Model artifacts not found; run training first.")

    model = joblib.load(model_path)
    transformer = SaaSFeatureTransformer.load(trans_path)
    return model, transformer


def test_shap_explainer_initialization_and_attribution(artifacts):
    """Test that SHAP explainer extracts top 3 drivers with valid metadata."""
    model, transformer = artifacts
    explainer = SaaSChurnExplainer(model, transformer.feature_names_)

    # Create distressed account payload
    high_risk_raw = {
        "account_id": "ACC-TEST-01",
        "contract_tier": "Enterprise",
        "contract_duration_months": 12,
        "tenure_months": 6,
        "monthly_recurring_revenue": 22000.0,
        "licensed_seats": 300,
        "active_users_last_30d": 45,
        "seat_utilization_ratio": 0.15,  # severe underutilization
        "login_frequency_last_30d": 2.0,
        "login_decay_ratio": 0.35,  # severe decay
        "feature_adoption_score": 0.20,
        "support_tickets_last_90d": 14,
        "p1_tickets_last_30d": 4,  # critical outages
        "ticket_escalation_velocity": 3.0,
        "csm_touchpoints_last_90d": 0,
        "billing_overdue_days": 30,
        "nps_score": -45.0,
    }

    df_raw = pd.DataFrame([high_risk_raw])
    transformed = transformer.transform(df_raw)

    drivers = explainer.explain_instance(
        transformed_features=transformed,
        raw_inputs=high_risk_raw,
        top_k=3,
    )

    assert len(drivers) == 3
    for d in drivers:
        assert "feature" in d
        assert "display_name" in d
        assert "shap_value" in d
        assert "raw_value" in d
        assert "direction" in d
        assert d["direction"] in ["INCREASES_CHURN_RISK", "PROTECTS_ACCOUNT"]
        assert "actionable_recommendation" in d
        assert len(d["actionable_recommendation"]) > 10

    # For this severely distressed account, the top driver should increase churn risk
    assert drivers[0]["direction"] == "INCREASES_CHURN_RISK"
