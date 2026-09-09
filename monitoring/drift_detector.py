"""
Evidently AI & Statistical Data Drift Detection Suite for B2B SaaS Telemetry.

Monitors covariate shift across production inference windows against baseline training data:
- Continuous behavioral features: Two-Sample Kolmogorov-Smirnov (KS) test (p < 0.05)
- Categorical / discrete distributions: Population Stability Index (PSI > 0.10)
- Generates interactive Evidently HTML drift reports and structured JSON alerts.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_psi(
    reference: Union[np.ndarray, pd.Series],
    current: Union[np.ndarray, pd.Series],
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate the Population Stability Index (PSI) between reference and current populations.

    Thresholds:
    """
    ref = np.asarray(reference.dropna()) if hasattr(reference, "dropna") else np.asarray(reference)
    cur = np.asarray(current.dropna()) if hasattr(current, "dropna") else np.asarray(current)

    ref = ref[~pd.isna(ref)]
    cur = cur[~pd.isna(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # For categorical / discrete values, compute direct value frequencies
    if ref.dtype == object or str(ref.dtype).startswith("str") or len(np.unique(ref)) <= 5:
        all_cats = np.unique(np.concatenate([ref, cur]))
        ref_counts = pd.Series(ref).value_counts(normalize=True).reindex(all_cats, fill_value=0.0)
        cur_counts = pd.Series(cur).value_counts(normalize=True).reindex(all_cats, fill_value=0.0)

        ref_pct = np.clip(ref_counts.values, epsilon, 1.0)
        cur_pct = np.clip(cur_counts.values, epsilon, 1.0)
        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return round(float(psi_val), 4)

    # For continuous features, quantile-based binning on reference
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref), epsilon, 1.0)
    cur_pct = np.clip(cur_counts / len(cur), epsilon, 1.0)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi_val), 4)


def calculate_ks_test(
    reference: Union[np.ndarray, pd.Series],
    current: Union[np.ndarray, pd.Series],
) -> Tuple[float, float]:
    """
    Compute Two-Sample Kolmogorov-Smirnov test for continuous features.
    Returns (ks_statistic, p_value).
    """
    ref = np.asarray(reference.dropna()) if hasattr(reference, "dropna") else np.asarray(reference)
    cur = np.asarray(current.dropna()) if hasattr(current, "dropna") else np.asarray(current)

    ref = ref[~pd.isna(ref)]
    cur = cur[~pd.isna(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0

    res = ks_2samp(ref, cur)
    return round(float(res.statistic), 4), round(float(res.pvalue), 6)


class SaaSDriftDetector:
    """
    Production drift detection engine for continuous behavioral SaaS telemetry.
    Runs KS-tests and PSI against the training baseline and generates diagnostic reports.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.10,
    ):
        self.reference_df = reference_data.copy()
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold

        # Segregate columns
        self.categorical_cols = [
            col for col in ["contract_tier", "contract_duration_months"]
            if col in self.reference_df.columns
        ]
        self.continuous_cols = [
            col for col in self.reference_df.select_dtypes(include=[np.number]).columns
            if col not in self.categorical_cols and col not in ["churn", "account_id"]
        ]
        logger.info(
            "Initialized SaaSDriftDetector: %d continuous features (KS-test alpha=%.2f), "
            "%d categorical features (PSI threshold=%.2f)",
            len(self.continuous_cols), self.ks_alpha,
            len(self.categorical_cols), self.psi_threshold,
        )

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate incoming inference batch against baseline distributions.
        """
        drifted_features = []
        feature_summaries = {}

        # 1. Continuous Behavioral Features (Two-Sample KS-Test)
        for col in self.continuous_cols:
            if col in current_data.columns:
                ref_series = self.reference_df[col]
                cur_series = current_data[col]

                ks_stat, p_val = calculate_ks_test(ref_series, cur_series)
                has_drifted = p_val < self.ks_alpha

                feature_summaries[col] = {
                    "test_type": "Two-Sample Kolmogorov-Smirnov",
                    "statistic": ks_stat,
                    "p_value": p_val,
                    "threshold": self.ks_alpha,
                    "drift_detected": has_drifted,
                    "ref_mean": round(float(ref_series.mean()), 3),
                    "cur_mean": round(float(cur_series.mean()), 3),
                }

                if has_drifted:
                    drifted_features.append({
                        "feature": col,
                        "test": "KS-Test",
                        "p_value": p_val,
                        "statistic": ks_stat,
                        "threshold": self.ks_alpha,
                        "ref_mean": round(float(ref_series.mean()), 3),
                        "cur_mean": round(float(cur_series.mean()), 3),
                    })

        # 2. Categorical & Discrete Features (PSI)
        for col in self.categorical_cols:
            if col in current_data.columns:
                ref_series = self.reference_df[col]
                cur_series = current_data[col]

                psi_score = calculate_psi(ref_series, cur_series)
                has_drifted = psi_score >= self.psi_threshold

                feature_summaries[col] = {
                    "test_type": "Population Stability Index (PSI)",
                    "statistic": psi_score,
                    "threshold": self.psi_threshold,
                    "drift_detected": has_drifted,
                }

                if has_drifted:
                    drifted_features.append({
                        "feature": col,
                        "test": "PSI",
                        "statistic": psi_score,
                        "threshold": self.psi_threshold,
                    })

        total_features = len(feature_summaries)
        drift_count = len(drifted_features)
        drift_share = round(drift_count / max(total_features, 1), 4)
        overall_drift = drift_count > 0

        logger.info(
            "Drift analysis complete: %d / %d features drifted (share: %.2f%%). Overall drift: %s",
            drift_count, total_features, drift_share * 100, overall_drift,
        )

        return {
            "drift_detected": overall_drift,
            "total_features": total_features,
            "drifted_features_count": drift_count,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "feature_details": feature_summaries,
        }

    def generate_evidently_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "monitoring/reports/drift_report.html",
    ) -> Optional[str]:
        """
        Generate and save an interactive Evidently AI visual drift report HTML file.
        """
        try:
            from evidently import Report
            from evidently.legacy.metric_preset import DataDriftPreset

            cols_to_use = [
                c for c in self.reference_df.columns
                if c in current_data.columns and c not in ["churn", "account_id"]
            ]

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_df[cols_to_use],
                current_data=current_data[cols_to_use],
            )

            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(out_file))
            logger.info("Saved Evidently visual HTML report to %s", out_file)
            return str(out_file)
        except Exception as e:
            logger.warning("Evidently report generation skipped: %s", str(e))
            return None
