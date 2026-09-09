"""
Model Training, Validation, and Calibration Pipeline for B2B SaaS Churn Prediction.

Trains LightGBM classifier with:
- Scale_pos_weight calibration for class imbalance
- PR-AUC optimization (target >= 0.84)
- Precision/Recall trade-off & F1 threshold calibration
- Benchmark comparison against baseline
- Full model and feature artifact serialization
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.generate_telemetry import generate_synthetic_telemetry
from src.features.build_features import prepare_training_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for imbalanced classification."""
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Search for the decision threshold that maximizes F1 score on validation predictions.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_threshold = 0.5
    best_score = -1.0

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(th)

    logger.info("Optimal decision threshold found: %.3f (F1: %.4f)", best_threshold, best_score)
    return best_threshold, best_score


def train_pipeline(
    data_path: Optional[str] = None,
    artifacts_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run end-to-end training, benchmarking, calibration, and serialization.
    """
    # 1. Load or generate dataset
    if data_path and Path(data_path).exists():
        logger.info("Loading telemetry dataset from %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.info("No existing dataset provided. Generating synthetic telemetry...")
        df = generate_synthetic_telemetry(n_samples=10000, random_state=random_state)
        sample_path = Path("data/sample_telemetry.csv")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sample_path, index=False)
        logger.info("Saved sample telemetry to %s", sample_path)

    # 2. Strict featurization ordering (split before fitting)
    X_train, X_test, y_train, y_test, transformer = prepare_training_splits(
        df, test_size=0.20, random_state=random_state
    )

    # 3. Calculate scale_pos_weight for class imbalance
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)
    logger.info(
        "Class distribution: Neg=%d, Pos=%d, scale_pos_weight=%.2f",
        negative_count,
        positive_count,
        scale_pos_weight,
    )

    # 4. Benchmark ML Baselines
    # Naive dummy baseline
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = evaluate_predictions(y_test, dummy_probs, 0.5)
    logger.info(
        "Dummy Baseline PR-AUC: %.4f | ROC-AUC: %.4f",
        dummy_metrics["pr_auc"],
        dummy_metrics["roc_auc"],
    )

    # Scaled Logistic Regression baseline
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_probs, 0.5)
    logger.info(
        "Logistic Regression Baseline PR-AUC: %.4f | ROC-AUC: %.4f",
        lr_metrics["pr_auc"],
        lr_metrics["roc_auc"],
    )

    # 5. High-Impact LightGBM Classifier
    lgbm = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight * 0.8,  # Calibrated imbalance weighting
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    logger.info("Fitting LightGBM classifier...")
    lgbm.fit(X_train, y_train)

    # Evaluate on Train and Test
    train_probs = lgbm.predict_proba(X_train)[:, 1]
    test_probs = lgbm.predict_proba(X_test)[:, 1]

    # Find optimal threshold using training/validation predictions
    optimal_th, _ = find_optimal_threshold(y_train, train_probs)

    train_metrics = evaluate_predictions(y_train, train_probs, optimal_th)
    test_metrics = evaluate_predictions(y_test, test_probs, optimal_th)

    logger.info("=== LightGBM Test Performance ===")
    logger.info("PR-AUC:    %.4f (Target >= 0.84)", test_metrics["pr_auc"])
    logger.info("ROC-AUC:   %.4f", test_metrics["roc_auc"])
    logger.info("F1 Score:  %.4f at threshold %.3f", test_metrics["f1"], optimal_th)
    logger.info("Precision: %.4f", test_metrics["precision"])
    logger.info("Recall:    %.4f", test_metrics["recall"])
    logger.info("Brier:     %.4f", test_metrics["brier_score"])

    # Verify acceptance criteria
    if test_metrics["pr_auc"] < 0.84:
        logger.warning(
            "PR-AUC %.4f is below target 0.84. Re-tuning required.", test_metrics["pr_auc"]
        )
    else:
        logger.info(
            "SUCCESS: PR-AUC %.4f meets production target (>= 0.84).", test_metrics["pr_auc"]
        )

    # Confusion matrix
    y_test_pred = (test_probs >= optimal_th).astype(int)
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    logger.info("Test Confusion Matrix:\n%s", confusion_matrix(y_test, y_test_pred))

    # 6. Save Artifacts
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_file = artifacts_path / "lgbm_model.joblib"
    transformer_file = artifacts_path / "feature_transformer.joblib"
    metadata_file = artifacts_path / "metadata.json"

    joblib.dump(lgbm, model_file)
    transformer.save(transformer_file)

    metadata = {
        "model_version": "1.0.0",
        "model_type": "LightGBMClassifier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimal_threshold": optimal_th,
        "feature_names": transformer.feature_names_,
        "raw_numerical_cols": list(X_train.columns),
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "baselines": {
            "dummy": dummy_metrics,
            "logistic_regression": lr_metrics,
        },
        "confusion_matrix": cm,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "churn_rate_test": round(float(y_test.mean()), 4),
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Serialized model to %s", model_file)
    logger.info("Serialized transformer to %s", transformer_file)
    logger.info("Saved metadata to %s", metadata_file)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train SaaS Churn Early-Warning Model.")
    parser.add_argument(
        "--data", type=str, default="data/sample_telemetry.csv", help="Dataset path"
    )
    parser.add_argument("--artifacts", type=str, default="models", help="Artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_pipeline(data_path=args.data, artifacts_dir=args.artifacts, random_state=args.seed)


if __name__ == "__main__":
    main()
