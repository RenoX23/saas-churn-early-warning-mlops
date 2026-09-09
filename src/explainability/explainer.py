"""
SHAP Explainability Module for B2B SaaS Churn Early-Warning.

Provides low-latency local feature attribution using shap.TreeExplainer,
extracting the top actionable risk drivers with directionality, magnitude,
and Customer Success intervention playbooks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Human-readable labels and playbooks for Customer Success intervention
FEATURE_METADATA_MAP: Dict[str, Dict[str, str]] = {
    "seat_utilization_ratio": {
        "display_name": "Active Seat Utilization",
        "action_increase": (
            "Account is underutilizing paid licenses. Schedule an adoption workshop."
        ),
        "action_decrease": "Healthy seat adoption across business units.",
    },
    "login_decay_ratio": {
        "display_name": "30-Day Login Velocity Decay",
        "action_increase": (
            "Sharp dropoff in daily user engagement. Trigger executive sponsor check-in."
        ),
        "action_decrease": "Engagement velocity is stable or expanding.",
    },
    "p1_tickets_last_30d": {
        "display_name": "Critical P1 Support Escalations",
        "action_increase": (
            "Unresolved P1 outages blocking core workflows. Escalate to VP of Engineering."
        ),
        "action_decrease": "Zero critical system escalations.",
    },
    "ticket_escalation_velocity": {
        "display_name": "Support Escalation Spike Velocity",
        "action_increase": "Support ticket volume spiking far above normal run rate.",
        "action_decrease": "Support ticket volume within normal operating limits.",
    },
    "billing_overdue_days": {
        "display_name": "Invoice Arrears & Billing Overdue",
        "action_increase": "Unpaid invoices past grace period. Engage finance and procurement.",
        "action_decrease": "Account billing is in good standing.",
    },
    "feature_adoption_score": {
        "display_name": "Platform Feature Adoption Breadth",
        "action_increase": (
            "Low product breadth usage. Customer has not activated core integrations."
        ),
        "action_decrease": "Deep sticky product adoption with multiple enabled workflows.",
    },
    "csm_touchpoints_last_90d": {
        "display_name": "Customer Success Engagement Frequency",
        "action_increase": "Account has been neglected with minimal CSM touchpoints.",
        "action_decrease": "Active cadence of quarterly business reviews and touchpoints.",
    },
    "nps_score_imputed": {
        "display_name": "Net Promoter Score (Customer Sentiment)",
        "action_increase": (
            "Detractor sentiment or negative CSAT recorded. Conduct immediate post-mortem."
        ),
        "action_decrease": "Promoter sentiment indicating high account satisfaction.",
    },
    "contract_duration_months": {
        "display_name": "Contract Term Commitment",
        "action_increase": "Short contract duration nearing renewal window.",
        "action_decrease": "Multi-year enterprise commitment providing strong retention buffer.",
    },
    "underutilization_risk": {
        "display_name": "Critical Seat Underutilization Flag",
        "action_increase": "Account utilization has dropped below 50% threshold.",
        "action_decrease": "Seat utilization healthy.",
    },
    "steep_login_decay": {
        "display_name": "Severe Login Decay Anomaly",
        "action_increase": "30-day login frequency dropped by more than 30%.",
        "action_decrease": "Login trends normal.",
    },
    "p1_ticket_density": {
        "display_name": "P1 Escalation Density per Active Seat",
        "action_increase": "High concentration of critical tickets per active user.",
        "action_decrease": "Low escalation density.",
    },
    "high_mrr_neglect_flag": {
        "display_name": "High-MRR Account Neglect Risk",
        "action_increase": (
            "High revenue tier account ($10k+/mo) with fewer than 2 CSM interactions."
        ),
        "action_decrease": "Enterprise account adequately serviced.",
    },
    "delinquent_flag": {
        "display_name": "Severe Payment Delinquency Flag",
        "action_increase": "Billing past due >15 days. Imminent service suspension risk.",
        "action_decrease": "No severe payment arrears.",
    },
}


@dataclass
class RiskDriver:
    """Individual feature contribution to churn prediction."""

    feature_name: str
    display_name: str
    shap_value: float
    raw_value: Any
    direction: str  # 'INCREASES_CHURN_RISK' or 'PROTECTS_ACCOUNT'
    recommendation: str


class SaaSChurnExplainer:
    """
    Sub-millisecond SHAP explanation engine built on LightGBM TreeExplainer.
    Pre-allocates the explainer to guarantee sub-35ms end-to-end API latency.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        logger.info("Initializing SHAP TreeExplainer for %d features...", len(feature_names))
        # TreeExplainer is fast for tree-based models
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(
        self,
        transformed_features: pd.DataFrame,
        raw_inputs: Dict[str, Any],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Compute local SHAP waterfall attribution for a single instance
        and return the top K actionable drivers.
        """
        # Calculate raw SHAP values (log-odds space)
        shap_values = self.explainer.shap_values(transformed_features)

        # For binary classification, TreeExplainer returns array of shape (N, D)
        # or list of [neg_shap, pos_shap]
        if isinstance(shap_values, list):
            # Binary class: pos_shap is index 1
            instance_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            # Shape (N, D, 2)
            instance_shap = shap_values[0, :, 1]
        else:
            instance_shap = shap_values[0]

        # Pair features with their SHAP values and raw values
        drivers: List[RiskDriver] = []
        for feat_name, s_val in zip(self.feature_names, instance_shap):
            raw_val = raw_inputs.get(feat_name, transformed_features.iloc[0].get(feat_name, 0.0))
            # Format raw value for clean JSON presentation
            if isinstance(raw_val, float):
                raw_val = round(raw_val, 4)

            meta = FEATURE_METADATA_MAP.get(
                feat_name,
                {
                    "display_name": feat_name.replace("_", " ").title(),
                    "action_increase": "Review account engagement metrics.",
                    "action_decrease": "Feature contributes positively to account health.",
                },
            )

            direction = "INCREASES_CHURN_RISK" if s_val > 0 else "PROTECTS_ACCOUNT"
            recommendation = meta["action_increase"] if s_val > 0 else meta["action_decrease"]

            drivers.append(
                RiskDriver(
                    feature_name=feat_name,
                    display_name=meta["display_name"],
                    shap_value=round(float(s_val), 4),
                    raw_value=raw_val,
                    direction=direction,
                    recommendation=recommendation,
                )
            )

        # Sort primarily by features that actively increase churn risk (positive SHAP descending)
        # If fewer than top_k positive risk drivers, fill with largest absolute magnitude drivers
        risk_increasing = [d for d in drivers if d.shap_value > 0]
        risk_increasing.sort(key=lambda d: d.shap_value, reverse=True)

        if len(risk_increasing) >= top_k:
            selected_drivers = risk_increasing[:top_k]
        else:
            # Include protective features if positive drivers are scarce
            remaining = [d for d in drivers if d not in risk_increasing]
            remaining.sort(key=lambda d: abs(d.shap_value), reverse=True)
            selected_drivers = (risk_increasing + remaining)[:top_k]

        return [
            {
                "feature": d.feature_name,
                "display_name": d.display_name,
                "shap_value": d.shap_value,
                "raw_value": d.raw_value,
                "direction": d.direction,
                "actionable_recommendation": d.recommendation,
            }
            for d in selected_drivers
        ]
