"""
Unit tests for SaaS telemetry data generation and feature engineering pipeline.
"""

import pandas as pd

from src.data.generate_telemetry import generate_synthetic_telemetry
from src.features.build_features import (
    CATEGORICAL_COLS,
    RAW_NUMERICAL_COLS,
    SaaSFeatureTransformer,
    prepare_training_splits,
)


def test_telemetry_generation():
    """Verify synthetic dataset generation integrity and class distribution."""
    df = generate_synthetic_telemetry(n_samples=500, random_state=42)
    assert len(df) == 500
    assert "churn" in df.columns
    assert "account_id" in df.columns

    # Verify realistic imbalanced churn rate (~4% - 10%)
    churn_rate = df["churn"].mean()
    assert 0.03 <= churn_rate <= 0.12, f"Unexpected churn rate: {churn_rate}"

    # Verify required raw columns
    for col in RAW_NUMERICAL_COLS:
        assert col in df.columns, f"Missing expected column: {col}"
    for col in CATEGORICAL_COLS:
        assert col in df.columns, f"Missing expected categorical: {col}"

    # Verify missingness in NPS is present
    assert df["nps_score"].isna().sum() > 0, "NPS should simulate realistic missingness"


def test_feature_transformer_fit_transform():
    """Verify feature engineering produces expected features and zero leakage."""
    df = generate_synthetic_telemetry(n_samples=600, random_state=123)
    X_train, X_test, y_train, y_test, transformer = prepare_training_splits(
        df, test_size=0.25, random_state=123
    )

    assert len(X_train) == 450
    assert len(X_test) == 150
    assert len(y_train) == 450
    assert len(y_test) == 150

    # Ensure no NaN values remain after transformation
    assert not X_train.isna().any().any(), "X_train contains unexpected NaNs"
    assert not X_test.isna().any().any(), "X_test contains unexpected NaNs"

    # Verify engineered behavioral features exist
    expected_engineered = [
        "underutilization_risk",
        "steep_login_decay",
        "p1_ticket_density",
        "high_mrr_neglect_flag",
        "delinquent_flag",
        "nps_missing_flag",
        "nps_score_imputed",
    ]
    for feat in expected_engineered:
        assert feat in X_train.columns, f"Engineered feature {feat} not in columns"

    # Verify exact column match between train and test
    assert list(X_train.columns) == list(X_test.columns)


def test_transformer_serialization(tmp_path):
    """Verify transformer can be saved and loaded with identical output."""
    df = generate_synthetic_telemetry(n_samples=200, random_state=42)
    transformer = SaaSFeatureTransformer()
    transformer.fit(df)

    save_path = tmp_path / "transformer.joblib"
    transformer.save(save_path)

    loaded = SaaSFeatureTransformer.load(save_path)
    assert loaded.feature_names_ == transformer.feature_names_
    assert loaded.median_nps == transformer.median_nps

    out_original = transformer.transform(df.head(5))
    out_loaded = loaded.transform(df.head(5))
    pd.testing.assert_frame_equal(out_original, out_loaded)
