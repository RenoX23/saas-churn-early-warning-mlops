"""
Pydantic v2 data schemas and validation models for the SaaS Churn Early-Warning API.
"""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AccountTelemetryInput(BaseModel):
    """Raw account behavioral telemetry for single inference."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "account_id": "ACC-10042",
                "contract_tier": "Enterprise",
                "contract_duration_months": 12,
                "tenure_months": 9,
                "monthly_recurring_revenue": 18500.0,
                "licensed_seats": 250,
                "active_users_last_30d": 60,
                "seat_utilization_ratio": 0.24,
                "login_frequency_last_30d": 3.1,
                "login_decay_ratio": 0.42,
                "feature_adoption_score": 0.38,
                "support_tickets_last_90d": 12,
                "p1_tickets_last_30d": 3,
                "ticket_escalation_velocity": 2.6,
                "csm_touchpoints_last_90d": 1,
                "billing_overdue_days": 22,
                "nps_score": -15.0,
            }
        },
    )

    account_id: str = Field(..., description="Unique enterprise account identifier")
    contract_tier: Literal["Enterprise", "Mid-Market", "SMB"] = Field(
        ..., description="Account subscription tier"
    )
    contract_duration_months: int = Field(
        ..., ge=1, le=60, description="Contract term commitment in months"
    )
    tenure_months: int = Field(..., ge=1, le=120, description="Customer relationship age in months")
    monthly_recurring_revenue: float = Field(
        ..., ge=0.0, description="Contract Monthly Recurring Revenue (MRR) in USD"
    )
    licensed_seats: int = Field(..., ge=1, description="Total contracted/paid seats")
    active_users_last_30d: int = Field(
        ..., ge=0, description="Distinct active users logged in over past 30 days"
    )
    seat_utilization_ratio: float = Field(
        ..., ge=0.0, le=2.0, description="Active seats divided by licensed capacity"
    )
    login_frequency_last_30d: float = Field(
        ..., ge=0.0, description="Average logins per active seat in past 30 days"
    )
    login_decay_ratio: float = Field(
        ..., ge=0.0, description="Ratio of recent 30d login frequency to prior 30d baseline"
    )
    feature_adoption_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized platform breadth adoption index (0 to 1)"
    )
    support_tickets_last_90d: int = Field(
        ..., ge=0, description="Total support cases opened in past 90 days"
    )
    p1_tickets_last_30d: int = Field(
        ..., ge=0, description="Critical severity (P1) incidents logged in past 30 days"
    )
    ticket_escalation_velocity: float = Field(
        ..., ge=0.0, description="Rate of critical tickets relative to total run rate"
    )
    csm_touchpoints_last_90d: int = Field(
        ..., ge=0, description="CSM quarterly business reviews or proactively scheduled meetings"
    )
    billing_overdue_days: int = Field(
        ..., ge=0, description="Days outstanding past invoice payment terms"
    )
    nps_score: Optional[float] = Field(
        None, ge=-100.0, le=100.0, description="Net Promoter Score (-100 to +100), if responded"
    )


class RiskDriverSchema(BaseModel):
    """Actionable feature attribution returned by SHAP explainer."""

    feature: str = Field(..., description="Canonical model feature identifier")
    display_name: str = Field(..., description="Human-readable feature title")
    shap_value: float = Field(..., description="SHAP attribution value (log-odds impact)")
    raw_value: Any = Field(..., description="Original input feature value")
    direction: Literal["INCREASES_CHURN_RISK", "PROTECTS_ACCOUNT"] = Field(
        ..., description="Direction of feature impact on churn probability"
    )
    actionable_recommendation: str = Field(
        ..., description="Prescriptive action for Customer Success intervention"
    )


class PredictionResponse(BaseModel):
    """Real-time single-account churn prediction payload."""

    account_id: str
    churn_probability: float
    predicted_churn: bool
    risk_level: Literal["CRITICAL", "ELEVATED", "NOMINAL"]
    optimal_threshold: float
    top_risk_drivers: List[RiskDriverSchema]
    inference_latency_ms: float
    contract_tier: Optional[str] = None
    monthly_recurring_revenue: Optional[float] = None


class BatchTelemetryInput(BaseModel):
    """Batch CRM account scoring payload."""

    accounts: List[AccountTelemetryInput] = Field(
        ..., min_length=1, max_length=1000, description="List of account telemetry records"
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated batch prediction response."""

    total_accounts: int
    at_risk_count: int
    mean_churn_probability: float
    total_latency_ms: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """System liveness and readiness probe response."""

    status: str
    model_version: str
    model_type: str
    optimal_threshold: float
    features_count: int
    uptime_seconds: float
