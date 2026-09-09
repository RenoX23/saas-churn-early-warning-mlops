"""
Unit and integration tests for LightGBM model training, PR-AUC, and serialization.
"""

import json
from pathlib import Path

import joblib

from src.models.train import train_pipeline


def test_train_pipeline_and_artifacts(tmp_path):
    """Run pipeline on small sample and verify artifacts and performance."""
    artifacts_dir = str(tmp_path / "models")

    metadata = train_pipeline(
        data_path=None,  # will generate synthetic data
        artifacts_dir=artifacts_dir,
        random_state=42,
    )

    # 1. Check metric keys and thresholds
    assert "test_metrics" in metadata
    metrics = metadata["test_metrics"]
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "threshold" in metrics

    # PR-AUC should comfortably exceed target of 0.84 on the synthetic enterprise data
    assert metrics["pr_auc"] >= 0.84, f"PR-AUC {metrics['pr_auc']} below required 0.84"
    assert 0.1 <= metrics["threshold"] <= 0.9, f"Invalid threshold: {metrics['threshold']}"

    # 2. Check serialized files exist
    model_path = Path(artifacts_dir) / "lgbm_model.joblib"
    transformer_path = Path(artifacts_dir) / "feature_transformer.joblib"
    metadata_path = Path(artifacts_dir) / "metadata.json"

    assert model_path.exists()
    assert transformer_path.exists()
    assert metadata_path.exists()

    # 3. Test deserialization and inference
    loaded_model = joblib.load(model_path)
    loaded_transformer = joblib.load(transformer_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)

    assert loaded_meta["model_version"] == "1.0.0"

    # Score a single test record
    dummy_input = {
        "contract_tier": "Enterprise",
        "contract_duration_months": 12,
        "tenure_months": 8,
        "monthly_recurring_revenue": 15000.0,
        "licensed_seats": 200,
        "active_users_last_30d": 50,  # 25% utilization (high risk)
        "seat_utilization_ratio": 0.25,
        "login_frequency_last_30d": 3.2,
        "login_decay_ratio": 0.45,  # steep decay
        "feature_adoption_score": 0.35,
        "support_tickets_last_90d": 10,
        "p1_tickets_last_30d": 3,  # escalated
        "ticket_escalation_velocity": 2.5,
        "csm_touchpoints_last_90d": 1,
        "billing_overdue_days": 25,
        "nps_score": -20.0,
    }
    import pandas as pd

    df_sample = pd.DataFrame([dummy_input])
    transformed = loaded_transformer.transform(df_sample)
    prob = loaded_model.predict_proba(transformed)[0, 1]

    # This high-risk enterprise account should have a high churn probability
    assert 0.0 <= prob <= 1.0
    assert prob > 0.50, f"Expected high risk, got probability {prob}"
