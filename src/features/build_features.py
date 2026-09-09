"""
Feature Engineering and Preprocessing Pipeline for B2B SaaS Churn Prediction.

Implements zero-leakage transformation, domain-specific behavioral feature derivations,
and serialization for production serving and SHAP interpretability.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical feature definitions
CATEGORICAL_COLS = ["contract_tier"]
RAW_NUMERICAL_COLS = [
    "contract_duration_months",
    "tenure_months",
    "monthly_recurring_revenue",
    "licensed_seats",
    "active_users_last_30d",
    "seat_utilization_ratio",
    "login_frequency_last_30d",
    "login_decay_ratio",
    "feature_adoption_score",
    "support_tickets_last_90d",
    "p1_tickets_last_30d",
    "ticket_escalation_velocity",
    "csm_touchpoints_last_90d",
    "billing_overdue_days",
    "nps_score",
]

TARGET_COL = "churn"
ID_COL = "account_id"


class SaaSFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Zero-leakage, production-grade feature engineering transformer.
    Learns statistical parameters (medians, categorical vocabularies) strictly on training data
    and applies deterministic transformations during inference.
    """

    def __init__(self):
        self.median_nps: float = 35.0
        self.tier_map: Dict[str, int] = {"SMB": 0, "Mid-Market": 1, "Enterprise": 2}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SaaSFeatureTransformer":
        """
        Fit transformer parameters strictly on training split.
        """
        df = X.copy()
        # Compute median NPS for imputation
        if "nps_score" in df.columns:
            self.median_nps = float(df["nps_score"].median(skipna=True))
            if np.isnan(self.median_nps):
                self.median_nps = 35.0

        # Determine output feature names by transforming a single dummy row
        transformed_dummy = self._engineer_features(df.head(2))
        self.feature_names_ = list(transformed_dummy.columns)
        self.is_fitted_ = True
        logger.info("Fitted SaaSFeatureTransformer with %d output features.", len(self.feature_names_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw account telemetry into engineered feature representation.
        """
        if not self.is_fitted_:
            raise RuntimeError("SaaSFeatureTransformer must be fitted before transforming data.")

        transformed = self._engineer_features(X)
        # Ensure exact column ordering matching training
        return transformed.reindex(columns=self.feature_names_, fill_value=0.0)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal feature engineering logic."""
        res = pd.DataFrame(index=df.index)

        # 1. Categorical Encodings (Ordinal mapping preserving tier maturity)
        tier_series = df["contract_tier"].astype(str)
        res["contract_tier_code"] = tier_series.map(self.tier_map).fillna(0).astype(int)

        # 2. Raw Numerical Features
        for col in RAW_NUMERICAL_COLS:
            if col in df.columns and col != "nps_score":
                res[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 3. Missingness & NPS Handling
        if "nps_score" in df.columns:
            nps = pd.to_numeric(df["nps_score"], errors="coerce")
            res["nps_missing_flag"] = nps.isna().astype(int)
            res["nps_score_imputed"] = nps.fillna(self.median_nps)
        else:
            res["nps_missing_flag"] = 1
            res["nps_score_imputed"] = self.median_nps

        # 4. Domain-Specific Behavioral Interaction Features
        # Underutilization risk: accounts utilizing < 50% of paid seats
        res["underutilization_risk"] = (res["seat_utilization_ratio"] < 0.50).astype(int)

        # Steep usage drop: 30d login decay > 30% reduction
        res["steep_login_decay"] = (res["login_decay_ratio"] < 0.70).astype(int)

        # Escalation intensity: P1 tickets per active user
        active_users = res["active_users_last_30d"].clip(lower=1)
        res["p1_ticket_density"] = np.round(res["p1_tickets_last_30d"] / active_users, 5)

        # Revenue-at-risk interaction: High MRR with low CSM touchpoints
        res["high_mrr_neglect_flag"] = (
            (res["monthly_recurring_revenue"] > 10000) & (res["csm_touchpoints_last_90d"] < 2)
        ).astype(int)

        # Severe billing delinquency flag (>15 days)
        res["delinquent_flag"] = (res["billing_overdue_days"] > 15).astype(int)

        return res

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize transformer to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved SaaSFeatureTransformer to %s", filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SaaSFeatureTransformer":
        """Load serialized transformer from disk."""
        return joblib.load(filepath)


def prepare_training_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SaaSFeatureTransformer]:
    """
    Split raw dataset into train and test sets BEFORE fitting transformer
    to ensure strict zero data leakage.

    Returns:
        X_train_trans, X_test_trans, y_train, y_test, transformer
    """
    from sklearn.model_selection import train_test_split

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL]

    # Stratified split to maintain rare churn class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Stratified train/test split: Train=%d, Test=%d (Churn Rate: %.2f%%)",
                len(y_train), len(y_test), y_train.mean() * 100)

    transformer = SaaSFeatureTransformer()
    # Fit strictly on train split
    transformer.fit(X_train, y_train)

    X_train_trans = transformer.transform(X_train)
    X_test_trans = transformer.transform(X_test)

    return X_train_trans, X_test_trans, y_train, y_test, transformer
