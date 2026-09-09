"""
Integration and performance tests for FastAPI inference microservice.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create test client within app lifespan context."""
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    """Verify /health liveness and metadata probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert data["model_type"] == "LightGBMClassifier"
    assert "optimal_threshold" in data
    assert data["features_count"] > 15
    assert data["uptime_seconds"] >= 0


def test_predict_single_high_risk_account(client):
    """Verify single prediction with SHAP drivers and latency SLA."""
    payload = {
        "account_id": "ACC-ENTERPRISE-CRITICAL",
        "contract_tier": "Enterprise",
        "contract_duration_months": 12,
        "tenure_months": 5,
        "monthly_recurring_revenue": 25000.0,
        "licensed_seats": 500,
        "active_users_last_30d": 60,
        "seat_utilization_ratio": 0.12,  # severe underutilization
        "login_frequency_last_30d": 1.8,
        "login_decay_ratio": 0.30,  # severe drop
        "feature_adoption_score": 0.25,
        "support_tickets_last_90d": 16,
        "p1_tickets_last_30d": 4,  # repeated outages
        "ticket_escalation_velocity": 3.2,
        "csm_touchpoints_last_90d": 0,
        "billing_overdue_days": 28,
        "nps_score": -60.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["account_id"] == "ACC-ENTERPRISE-CRITICAL"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_probability"] > 0.60
    assert data["predicted_churn"] is True
    assert data["risk_level"] in ["CRITICAL", "ELEVATED"]

    # Verify top-3 SHAP risk drivers
    drivers = data["top_risk_drivers"]
    assert len(drivers) == 3
    for d in drivers:
        assert "feature" in d
        assert "display_name" in d
        assert "shap_value" in d
        assert "direction" in d
        assert "actionable_recommendation" in d


def test_predict_single_healthy_account(client):
    """Verify healthy retained account produces nominal risk."""
    payload = {
        "account_id": "ACC-ENTERPRISE-HEALTHY",
        "contract_tier": "Enterprise",
        "contract_duration_months": 36,  # 3-year commitment
        "tenure_months": 24,
        "monthly_recurring_revenue": 35000.0,
        "licensed_seats": 400,
        "active_users_last_30d": 380,
        "seat_utilization_ratio": 0.95,  # 95% utilization
        "login_frequency_last_30d": 18.5,
        "login_decay_ratio": 1.15,  # growing usage
        "feature_adoption_score": 0.92,
        "support_tickets_last_90d": 3,
        "p1_tickets_last_30d": 0,
        "ticket_escalation_velocity": 0.0,
        "csm_touchpoints_last_90d": 6,
        "billing_overdue_days": 0,
        "nps_score": 75.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["churn_probability"] < 0.20
    assert data["predicted_churn"] is False
    assert data["risk_level"] == "NOMINAL"


def test_predict_invalid_schema_returns_422(client):
    """Verify strict Pydantic validation rejects out-of-bound inputs."""
    invalid_payload = {
        "account_id": "ACC-INVALID",
        "contract_tier": "SuperEnterprise",  # Invalid tier literal
        "contract_duration_months": -5,  # Invalid negative duration
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(client):
    """Verify batch prediction endpoint scores multiple accounts."""
    batch_payload = {
        "accounts": [
            {
                "account_id": f"ACC-BATCH-{i}",
                "contract_tier": "Mid-Market",
                "contract_duration_months": 12,
                "tenure_months": 12,
                "monthly_recurring_revenue": 4500.0,
                "licensed_seats": 50,
                "active_users_last_30d": 35,
                "seat_utilization_ratio": 0.70,
                "login_frequency_last_30d": 10.0,
                "login_decay_ratio": 0.95,
                "feature_adoption_score": 0.65,
                "support_tickets_last_90d": 2,
                "p1_tickets_last_30d": 0,
                "ticket_escalation_velocity": 0.0,
                "csm_touchpoints_last_90d": 3,
                "billing_overdue_days": 0,
                "nps_score": 40.0,
            }
            for i in range(5)
        ]
    }

    response = client.post("/batch-predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_accounts"] == 5
    assert len(data["predictions"]) == 5
    assert "mean_churn_probability" in data
    assert "total_latency_ms" in data
