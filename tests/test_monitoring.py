"""
Unit tests for data drift monitoring and alerting services.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from monitoring.alert_service import build_slack_alert_payload, send_drift_alert
from monitoring.drift_detector import SaaSDriftDetector, calculate_ks_test, calculate_psi
from monitoring.simulate_drift import inject_covariate_shift
from src.data.generate_telemetry import generate_synthetic_telemetry


def test_calculate_psi_stable():
    """Verify identical populations yield near-zero PSI."""
    np.random.seed(42)
    ref = np.random.normal(50, 10, 1000)
    cur = np.random.normal(50, 10, 1000)
    psi = calculate_psi(ref, cur)
    assert psi < 0.10, f"Expected stable PSI, got {psi}"


def test_calculate_psi_shifted():
    """Verify shifted populations breach 0.10 PSI threshold."""
    np.random.seed(42)
    ref = np.random.normal(50, 10, 1000)
    cur = np.random.normal(70, 10, 1000)  # substantial distribution shift
    psi = calculate_psi(ref, cur)
    assert psi >= 0.10, f"Expected drifted PSI, got {psi}"


def test_calculate_ks_test_stable():
    """Verify samples from same distribution fail to reject null hypothesis."""
    np.random.seed(42)
    ref = np.random.uniform(0, 1, 1000)
    cur = np.random.uniform(0, 1, 1000)
    stat, p_val = calculate_ks_test(ref, cur)
    assert p_val > 0.05, f"Expected non-significant KS p-value, got {p_val}"


def test_calculate_ks_test_shifted():
    """Verify shifted distribution produces p < 0.05."""
    np.random.seed(42)
    ref = np.random.uniform(0, 1, 1000)
    cur = np.random.uniform(0.5, 1.5, 1000)
    stat, p_val = calculate_ks_test(ref, cur)
    assert p_val < 0.05, f"Expected significant KS drift, got {p_val}"


def test_drift_detector_workflow():
    """Verify full drift detector detects injected covariate shift."""
    ref_df = generate_synthetic_telemetry(n_samples=500, random_state=42)
    detector = SaaSDriftDetector(ref_df, ks_alpha=0.05, psi_threshold=0.10)

    # 1. Clean current window
    clean_cur = generate_synthetic_telemetry(n_samples=500, random_state=101)
    clean_res = detector.detect_drift(clean_cur)
    # Drift share should be low on clean independent sample
    assert clean_res["drift_share"] < 0.20

    # 2. Shifted current window
    shifted_cur = inject_covariate_shift(clean_cur, shift_intensity=0.60)
    shifted_res = detector.detect_drift(shifted_cur)

    assert shifted_res["drift_detected"] is True
    assert shifted_res["drifted_features_count"] >= 3
    drifted_names = [d["feature"] for d in shifted_res["drifted_features"]]
    assert "login_decay_ratio" in drifted_names or "p1_tickets_last_30d" in drifted_names


def test_slack_alert_formatting_and_saving(tmp_path):
    """Verify alert payload generation and audit file persistence."""
    sample_results = {
        "drift_detected": True,
        "total_features": 16,
        "drifted_features_count": 4,
        "drift_share": 0.25,
        "drifted_features": [
            {
                "feature": "login_decay_ratio",
                "test": "KS-Test",
                "statistic": 0.65,
                "p_value": 0.00001,
                "ref_mean": 1.05,
                "cur_mean": 0.45,
            }
        ],
    }

    payload = build_slack_alert_payload(sample_results)
    assert "blocks" in payload
    assert len(payload["blocks"]) >= 3

    log_dir = str(tmp_path / "reports")
    success = send_drift_alert(sample_results, webhook_url=None, log_dir=log_dir)
    assert success is True
    assert (Path(log_dir) / "last_drift_alert.json").exists()
