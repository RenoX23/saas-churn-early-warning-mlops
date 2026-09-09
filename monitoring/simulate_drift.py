"""
Production Drift Simulation Runner.

Simulates synthetic distribution shift on enterprise SaaS account telemetry
(e.g., severe login decay and escalation spikes caused by an outage/release regression),
executes KS-test/PSI statistical evaluations, generates visual reports, and triggers alerts.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.alert_service import send_drift_alert
from monitoring.drift_detector import SaaSDriftDetector
from src.data.generate_telemetry import generate_synthetic_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def inject_covariate_shift(df: pd.DataFrame, shift_intensity: float = 0.50) -> pd.DataFrame:
    """
    Simulate realistic production covariate shift:
    - 30-day login frequency decay drops drastically (engagement collapse)
    - Support ticket escalations and P1 tickets spike (product outage/bug)
    - Active seat utilization ratio degrades
    """
    shifted = df.copy()
    logger.info("Injecting covariate shift (intensity: %.2f)...", shift_intensity)

    # 1. Acute login decay drop (users stopping daily use)
    shifted["login_decay_ratio"] = shifted["login_decay_ratio"] * (1.0 - shift_intensity * 0.7)

    # 2. Critical support escalation spike
    shifted["p1_tickets_last_30d"] = shifted["p1_tickets_last_30d"] + np.random.poisson(
        lam=2.5, size=len(shifted)
    )
    shifted["ticket_escalation_velocity"] = shifted["ticket_escalation_velocity"] * 2.2

    # 3. Seat utilization drop
    shifted["seat_utilization_ratio"] = np.clip(shifted["seat_utilization_ratio"] * 0.65, 0.05, 1.0)

    # 4. Shift in tier distribution (higher SMB proportion)
    tier_choices = ["Enterprise", "Mid-Market", "SMB"]
    shifted["contract_tier"] = np.random.choice(
        tier_choices, size=len(shifted), p=[0.10, 0.30, 0.60]
    )

    return shifted


def run_drift_simulation(
    reference_path: str = "data/sample_telemetry.csv",
    batch_size: int = 1000,
    simulate_shift: bool = True,
    generate_html: bool = True,
):
    """
    Run end-to-end drift detection workflow.
    """
    # 1. Load reference training baseline
    if Path(reference_path).exists():
        ref_df = pd.read_csv(reference_path)
    else:
        logger.info("Generating baseline reference data...")
        ref_df = generate_synthetic_telemetry(n_samples=5000, random_state=42)

    # 2. Generate current production inference window
    current_df = generate_synthetic_telemetry(n_samples=batch_size, random_state=999)
    if simulate_shift:
        current_df = inject_covariate_shift(current_df, shift_intensity=0.50)

    # 3. Execute Drift Analysis
    detector = SaaSDriftDetector(reference_data=ref_df, ks_alpha=0.05, psi_threshold=0.10)
    results = detector.detect_drift(current_df)

    logger.info("=== Statistical Drift Analysis Summary ===")
    logger.info("Drift Detected: %s", results["drift_detected"])
    logger.info(
        "Breached Features: %d / %d (%.1f%%)",
        results["drifted_features_count"],
        results["total_features"],
        results["drift_share"] * 100,
    )

    for item in results["drifted_features"]:
        p_val = item.get("p_value")
        p_str = f"p-val={p_val:.6f}" if p_val is not None else ""
        logger.warning(
            " -> Drifted Feature: %s [%s: stat=%.4f %s]",
            item["feature"],
            item["test"],
            item["statistic"],
            p_str,
        )

    # 4. Trigger Webhook Alerting
    if results["drift_detected"]:
        logger.info("Breach threshold exceeded. Dispatching webhook alert...")
        send_drift_alert(results)

    # 5. Optional Evidently HTML Visual Report
    if generate_html:
        detector.generate_evidently_html_report(
            current_data=current_df,
            output_path="monitoring/reports/drift_report.html",
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Simulate and detect telemetry drift.")
    parser.add_argument(
        "--data", type=str, default="data/sample_telemetry.csv", help="Reference baseline path"
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Inference batch size")
    parser.add_argument("--no-shift", action="store_true", help="Do not inject drift (test clean)")
    parser.add_argument("--no-html", action="store_true", help="Skip Evidently HTML report")
    args = parser.parse_args()

    run_drift_simulation(
        reference_path=args.data,
        batch_size=args.batch_size,
        simulate_shift=not args.no_shift,
        generate_html=not args.no_html,
    )


if __name__ == "__main__":
    main()
