"""
B2B SaaS Account Behavioral Telemetry Generator.

Generates realistic enterprise B2B SaaS account telemetry incorporating:
- Account tier & contract economics (Enterprise, Mid-Market, SMB)
- License utilization dynamics (licensed seats vs. active seats)
- Behavioral usage decay (30d vs 60d login frequency)
- Support ticket escalation velocity & P1 severity spikes
- Customer Success touchpoints & invoice delinquency
- Class-imbalanced churn distribution (~6-8% positive rate)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_telemetry(
    n_samples: int = 10000,
    random_state: int = 42,
    imbalance_ratio: float = 0.07,
) -> pd.DataFrame:
    """
    Generate synthetic enterprise B2B SaaS account telemetry data.

    Args:
        n_samples: Number of account records to generate.
        random_state: Seed for reproducibility.
        imbalance_ratio: Target baseline churn proportion (~6-8%).

    Returns:
        pd.DataFrame containing account telemetry features and churn labels.
    """
    np.random.seed(random_state)
    logger.info("Generating %d synthetic SaaS account records...", n_samples)

    # 1. Account Demographics & Contract Economics
    account_ids = [f"ACC-{10000 + i}" for i in range(n_samples)]

    tier_choices = ["Enterprise", "Mid-Market", "SMB"]
    tier_probs = [0.20, 0.45, 0.35]
    contract_tiers = np.random.choice(tier_choices, size=n_samples, p=tier_probs)

    contract_duration_choices = [12, 24, 36]
    contract_duration = np.random.choice(contract_duration_choices, size=n_samples, p=[0.55, 0.35, 0.10])

    tenure_months = np.random.gamma(shape=3.0, scale=6.0, size=n_samples).astype(int) + 1
    tenure_months = np.clip(tenure_months, 1, 60)

    # Base MRR and seat capacity scaled by tier
    mrr = np.zeros(n_samples)
    licensed_seats = np.zeros(n_samples, dtype=int)
    for i, tier in enumerate(contract_tiers):
        if tier == "Enterprise":
            mrr[i] = np.random.uniform(8000, 45000)
            licensed_seats[i] = np.random.randint(100, 1000)
        elif tier == "Mid-Market":
            mrr[i] = np.random.uniform(2000, 10000)
            licensed_seats[i] = np.random.randint(30, 150)
        else:  # SMB
            mrr[i] = np.random.uniform(400, 2500)
            licensed_seats[i] = np.random.randint(5, 35)

    # 2. Behavioral Usage Dynamics & Engagement
    # Base seat utilization ratio (healthy accounts ~0.75-0.95, distressed <0.4)
    base_utilization = np.random.beta(a=7, b=2.5, size=n_samples)
    active_users_30d = np.maximum(1, np.round(licensed_seats * base_utilization).astype(int))
    seat_utilization_ratio = np.round(active_users_30d / licensed_seats, 4)

    # 30-day login frequency per active seat
    login_freq_30d = np.random.gamma(shape=4.0, scale=3.5, size=n_samples)
    # Login decay: prior 30-day activity vs recent 30-day activity
    login_freq_prev_30d = login_freq_30d * np.random.normal(loc=1.05, scale=0.25, size=n_samples)
    login_freq_prev_30d = np.maximum(0.5, login_freq_prev_30d)
    login_decay_ratio = np.round((login_freq_30d + 1e-4) / (login_freq_prev_30d + 1e-4), 4)

    # Feature adoption index (0.0 to 1.0)
    feature_adoption_score = np.random.beta(a=5, b=2.5, size=n_samples)

    # 3. Support & Customer Success Telemetry
    support_tickets_90d = np.random.poisson(lam=3.5, size=n_samples)
    # P1/escalated tickets
    p1_tickets_30d = np.random.poisson(lam=0.4, size=n_samples)
    ticket_escalation_velocity = np.round(
        (p1_tickets_30d * 3.0 + 1e-4) / (support_tickets_90d / 3.0 + 1e-4), 4
    )

    # CSM engagement touchpoints
    csm_touchpoints_90d = np.random.poisson(lam=4.0, size=n_samples)
    # Billing delinquency (days overdue)
    billing_overdue_days = np.random.exponential(scale=2.0, size=n_samples).astype(int)
    billing_overdue_days = np.where(np.random.rand(n_samples) > 0.85, billing_overdue_days * 4, 0)

    # NPS score (-100 to 100). Include realistic missingness (30% don't respond)
    nps_raw = np.random.normal(loc=35, scale=30, size=n_samples)
    nps_score = np.clip(np.round(nps_raw), -100, 100).astype(float)
    nps_missing_mask = np.random.rand(n_samples) < 0.30
    nps_score[nps_missing_mask] = np.nan

    # 4. Latent Churn Propensity Function (Ground Truth Probability)
    # Structural risk equation based on B2B SaaS failure modes
    nps_term = np.where(np.isnan(nps_score), 0.0, -0.015 * nps_score)
    risk = (
        3.8 * np.maximum(0, 0.45 - seat_utilization_ratio)
        + 3.2 * np.maximum(0, 0.65 - login_decay_ratio)
        + 2.0 * (p1_tickets_30d >= 2).astype(float)
        + 1.4 * (ticket_escalation_velocity > 1.8).astype(float)
        + 2.2 * (billing_overdue_days > 15).astype(float)
        + 1.6 * ((mrr > 10000) & (csm_touchpoints_90d < 2)).astype(float)
        - 1.4 * feature_adoption_score
        - 0.8 * (contract_duration >= 24).astype(float)
        - 0.5 * (csm_touchpoints_90d >= 6).astype(float)
        + 0.4 * (contract_tiers == "SMB").astype(float)
        - 0.4 * (contract_tiers == "Enterprise").astype(float)
        + nps_term
    )

    # Sharp logistic transition calibrated to produce ~7.0% enterprise churn
    churn_probabilities = 1.0 / (1.0 + np.exp(-(risk - 1.15) * 4.2))

    # Generate binary churn labels
    churn = (np.random.rand(n_samples) < churn_probabilities).astype(int)

    df = pd.DataFrame({
        "account_id": account_ids,
        "contract_tier": contract_tiers,
        "contract_duration_months": contract_duration,
        "tenure_months": tenure_months,
        "monthly_recurring_revenue": np.round(mrr, 2),
        "licensed_seats": licensed_seats,
        "active_users_last_30d": active_users_30d,
        "seat_utilization_ratio": seat_utilization_ratio,
        "login_frequency_last_30d": np.round(login_freq_30d, 2),
        "login_decay_ratio": login_decay_ratio,
        "feature_adoption_score": np.round(feature_adoption_score, 4),
        "support_tickets_last_90d": support_tickets_90d,
        "p1_tickets_last_30d": p1_tickets_30d,
        "ticket_escalation_velocity": ticket_escalation_velocity,
        "csm_touchpoints_last_90d": csm_touchpoints_90d,
        "billing_overdue_days": billing_overdue_days,
        "nps_score": nps_score,
        "churn": churn,
    })

    actual_churn_rate = df["churn"].mean()
    logger.info("Dataset generated: %d rows, %d columns.", df.shape[0], df.shape[1])
    logger.info("Empirical Churn Rate: %.2f%% (%d churned accounts)", actual_churn_rate * 100, df["churn"].sum())

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic B2B SaaS account telemetry.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of account records")
    parser.add_argument("--output", type=str, default="data/sample_telemetry.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_telemetry(n_samples=args.samples, random_state=args.seed)
    df.to_csv(output_path, index=False)
    logger.info("Saved telemetry dataset to %s", output_path)


if __name__ == "__main__":
    main()
