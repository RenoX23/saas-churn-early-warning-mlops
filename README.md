# B2B SaaS Account Health & Churn Early-Warning Microservice with Drift Alerting

[![CI/CD Pipeline](https://github.com/RenoX23/saas-churn-early-warning-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/RenoX23/saas-churn-early-warning-mlops/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An active early-warning MLOps microservice that predicts enterprise B2B SaaS churn on imbalanced telemetry (0.88 PR-AUC), dynamically attributes top-3 actionable risk drivers via SHAP at sub-16ms latency, and dispatches automated Slack/Discord alerts upon statistical covariate shift (KS-test p < 0.05 / PSI > 0.10).**

---

## 1. Business Problem & Executive Framing

In enterprise subscription B2B SaaS, Customer Acquisition Costs (CAC) routinely exceed \$15,000 to \$50,000+ per account. When an enterprise account silently churns, the company suffers compounding Annual Recurring Revenue (ARR) leakage. 

Traditional churn prediction systems operate as **passive, offline batch jobs** (e.g. monthly scripts). By the time a batch job flags an account, the customer has typically already completed vendor procurement with a competitor and decided to terminate.

**What This System Solves**:
1. **Active Real-Time Early-Warning**: Scores account health continuously via REST API upon CRM and telemetry events rather than waiting for monthly batch cycles.
2. **Actionable Explainability at Inference**: Instead of returning an opaque probability, every prediction surfaces the **top 3 specific risk drivers** (direction, raw value, and Customer Success intervention playbooks) via SHAP `TreeExplainer`.
3. **Class-Imbalance Calibration**: Calibrated for realistic enterprise churn base rates (~5–7%), optimizing **PR-AUC (0.8837)** and decision thresholds ($\tau^* = 0.81$) to eliminate false alarms and prevent CS alert fatigue.
4. **Closed-Loop Drift Detection**: Monitors production inference windows against the training baseline using **Two-Sample Kolmogorov-Smirnov tests** ($p < 0.05$) and **Population Stability Index** ($\text{PSI} > 0.10$), dispatching real-time webhook alerts to Slack before model degradation impacts the business.

---

## 2. System Architecture & Flow

```
[B2B SaaS Account Telemetry: Usage Logs, Support Tickets, License Data]
                               │
                               ▼
[Zero-Leakage Feature Engineering Pipeline] (src/features/)
   ├── Behavioral Metrics: Seat utilization ratio, 30d login decay, ticket velocity
   └── Strict Featurization Ordering: Stratified split prior to fitting scalers/imputers
                               │
                               ▼
[Model Training & Experimentation Layer] (src/models/)
   ├── LightGBM Classifier (Scale_pos_weight & F1 threshold calibration)
   └── Model Serialization (lgbm_model.joblib, metadata.json)
                               │
                               ▼
[Containerized Inference Microservice (FastAPI + Docker)] (src/api/)
   ├── POST /predict        -> Single Account Risk Score + Top 3 SHAP Drivers (<16ms)
   ├── POST /batch-predict  -> Bulk CRM Account Scoring & Portfolio Analytics
   └── GET  /health         -> Liveness / Readiness Probes & Model Metadata
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[Explainability Layer (SHAP)]          [Continuous Drift Monitoring] (monitoring/)
   └── Pre-allocated TreeExplainer        ├── Continuous: Two-sample KS-Test (p < 0.05)
       extracting directionality &        └── Categorical: Population Stability Index (PSI > 0.1)
       Customer Success playbooks                    │
                                                     ▼ (Threshold breach)
                                        [Automated Slack / Discord Webhook Alert]
                                           └── Posts rich block alert detailing drifted features
```

---

## 3. Benchmark Performance & Evaluation

The model was evaluated against strict stratified holdout test splits (2,000 accounts) with an empirical churn base rate of 6.0%:

| Model / Baseline | PR-AUC | ROC-AUC | F1 Score | Precision | Recall | Brier Score | Latency (P50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stratified Dummy Baseline** | 0.0648 | 0.5244 | 0.1061 | 0.1040 | 0.1083 | 0.1095 | <1ms |
| **Balanced Logistic Regression** | 0.7876 | 0.9779 | 0.5441 | 0.3854 | **0.9250** | 0.0639 | ~3ms |
| **Production LightGBM (Ours)** | **0.8837** | **0.9883** | **0.7964** | **0.8713** | 0.7333 | **0.0189** | **15.57ms** |

* **Acceptance Criterion Met**: PR-AUC $\ge 0.84$ (**Achieved 0.8837**).
* **Serving Latency SLA**: Sub-35ms target (**Achieved P50: 15.57ms, P95: 16.65ms** with SHAP explainability enabled).
* **Test Confusion Matrix**:
  - True Negatives: **1,867** | False Positives: **13** (High precision prevents CS team alert fatigue)
  - False Negatives: **32** | True Positives: **88** (Captures early-stage enterprise churn risk)

---

## 4. Technical Stack & Architectural Decisions

| Layer | Tool / Technology | Architectural Rationale & Defense |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Modern typing, native performance, pre-built wheel availability. |
| **ML Engine** | LightGBM 4.7 | Fast gradient boosting over tabular data; native handling of non-linear behavioral interactions and missing values. |
| **Metric Focus** | PR-AUC ($\ge 0.84$) | In 6% imbalanced data, ROC-AUC is misleadingly inflated by true negatives. PR-AUC directly penalizes false alarms. |
| **Threshold Tuning** | F1 Calibration ($\tau^* = 0.81$) | Replaced naive 0.5 threshold with optimal search balancing precision (0.8713) against recall (0.7333). |
| **Explainability** | SHAP `TreeExplainer` | Tree-specific exact Shapley computation pre-allocated during startup to deliver local feature attribution in under 5ms. |
| **API Microservice** | FastAPI + Pydantic v2 | High-throughput async ASGI serving with strict schema validation and OpenAPI documentation. |
| **Containerization** | Multi-Stage Docker | Minimal runtime attack surface (builder stage separated from runtime) running as non-root user (`appuser`). |
| **CI/CD Pipeline** | GitHub Actions | Dual-stage automated testing: linting (`ruff`), test suite with coverage (`pytest`), and container build verification. |
| **Drift Monitoring** | Evidently AI & SciPy | Two-Sample KS-test ($p < 0.05$) for continuous features + PSI ($> 0.10$) for categorical distributions. |
| **Alerting** | Slack / Discord Webhooks | Formatted Slack Blocks JSON dispatching breached feature statistics, means, and remediation advice. |

---

## 5. Repository Structure

```
aiml-b2b-saas-churn-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI/CD pipeline (lint, test, docker build)
├── data/
│   ├── .gitkeep
│   └── sample_telemetry.csv       # 10,000 synthetic enterprise account records
├── docker/
│   └── Dockerfile                 # Multi-stage hardened production Dockerfile
├── models/
│   ├── feature_transformer.joblib # Serialized zero-leakage preprocessor
│   ├── lgbm_model.joblib          # Trained LightGBM classifier artifact
│   └── metadata.json              # Versioning, metrics, threshold, and feature signatures
├── monitoring/
│   ├── __init__.py
│   ├── alert_service.py           # Slack/Discord webhook alert dispatcher
│   ├── drift_detector.py          # KS-test & PSI statistical drift engine + Evidently
│   └── simulate_drift.py          # Production covariate shift simulation runner
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI microservice (/predict, /batch-predict, /health)
│   │   └── schemas.py             # Pydantic v2 request/response validation schemas
│   ├── data/
│   │   ├── __init__.py
│   │   └── generate_telemetry.py  # Enterprise B2B SaaS telemetry generator
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── explainer.py           # SHAP TreeExplainer & Customer Success playbooks
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py      # Zero-leakage behavioral feature engineering
│   └── models/
│       ├── __init__.py
│       └── train.py               # Model training, baseline benchmarking, calibration
├── tests/
│   ├── __init__.py
│   ├── test_api.py                # FastAPI endpoint integration & latency tests
│   ├── test_explainability.py     # SHAP TreeExplainer attribution unit tests
│   ├── test_features.py           # Feature transformer & telemetry unit tests
│   ├── test_model.py              # Model training & serialization tests
│   └── test_monitoring.py         # Statistical KS-test, PSI & webhook tests
├── .dockerignore                  # Docker build context exclusions
├── .gitignore                     # Git exclusion rules
├── docker-compose.yml             # Local Docker Compose service configuration
├── pyproject.toml                 # Packaging, ruff linter & pytest configurations
├── requirements.txt               # Pinned production and development dependencies
└── README.md                      # Production documentation
```

---

## 6. Quickstart & Installation

### Option A: Local Virtual Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/RenoX23/saas-churn-early-warning-mlops.git
cd saas-churn-early-warning-mlops

# 2. Create and activate Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies and project in editable mode
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Option B: Docker Containerization

```bash
# Build and run the multi-stage production container
docker compose up --build

# Microservice will be live at: http://localhost:8000
# OpenAPI Interactive Documentation: http://localhost:8000/docs
```

---

## 7. Execution Guide

### 1. Generate Telemetry & Train Model

```bash
# Generate 10,000 enterprise B2B SaaS account records
python src/data/generate_telemetry.py --samples 10000 --output data/sample_telemetry.csv

# Execute end-to-end training, baseline benchmarking, and threshold calibration
python src/models/train.py --data data/sample_telemetry.csv --artifacts models
```

### 2. Start the Inference Microservice

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Query the API

**Health Probe (`GET /health`)**:
```bash
curl -X GET http://localhost:8000/health
```
*Response*:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_type": "LightGBMClassifier",
  "optimal_threshold": 0.81,
  "features_count": 22,
  "uptime_seconds": 124.5
}
```

**Single Account Prediction with SHAP (`POST /predict`)**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC-10042",
    "contract_tier": "Enterprise",
    "contract_duration_months": 12,
    "tenure_months": 9,
    "monthly_recurring_revenue": 18500.0,
    "licensed_seats": 250,
    "active_users_last_30d": 55,
    "seat_utilization_ratio": 0.22,
    "login_frequency_last_30d": 2.8,
    "login_decay_ratio": 0.38,
    "feature_adoption_score": 0.35,
    "support_tickets_last_90d": 14,
    "p1_tickets_last_30d": 3,
    "ticket_escalation_velocity": 2.8,
    "csm_touchpoints_last_90d": 1,
    "billing_overdue_days": 24,
    "nps_score": -25.0
  }'
```
*Response Payload*:
```json
{
  "account_id": "ACC-10042",
  "churn_probability": 0.9412,
  "predicted_churn": true,
  "risk_level": "CRITICAL",
  "optimal_threshold": 0.81,
  "top_risk_drivers": [
    {
      "feature": "seat_utilization_ratio",
      "display_name": "Active Seat Utilization",
      "shap_value": 2.3412,
      "raw_value": 0.22,
      "direction": "INCREASES_CHURN_RISK",
      "actionable_recommendation": "Account is underutilizing paid licenses. Schedule an adoption workshop."
    },
    {
      "feature": "login_decay_ratio",
      "display_name": "30-Day Login Velocity Decay",
      "shap_value": 1.8741,
      "raw_value": 0.38,
      "direction": "INCREASES_CHURN_RISK",
      "actionable_recommendation": "Sharp dropoff in daily user engagement. Trigger executive sponsor check-in."
    },
    {
      "feature": "p1_tickets_last_30d",
      "display_name": "Critical P1 Support Escalations",
      "shap_value": 1.4523,
      "raw_value": 3,
      "direction": "INCREASES_CHURN_RISK",
      "actionable_recommendation": "Unresolved P1 outages blocking core workflows. Escalate to VP of Engineering."
    }
  ],
  "inference_latency_ms": 15.82
}
```

### 4. Run Automated Test Suite & Linter

```bash
# Run 16 comprehensive unit, integration, and latency tests with coverage
pytest --cov=src --cov-report=term-missing

# Run Ruff linter and style check
ruff check .
ruff format --check .
```

### 5. Simulate Production Covariate Shift & Webhook Alerting

```bash
# Simulate production telemetry shift and trigger Slack webhook alert
python monitoring/simulate_drift.py --batch-size 1000 --no-html

# View the generated Slack Blocks alert payload
cat monitoring/reports/last_drift_alert.json
```

---

## 8. Key Interview Defenses (Memorize Cold)

### Q1: Why did you prioritize PR-AUC over ROC-AUC?
> *"In enterprise B2B SaaS, churn is inherently rare (~5–8% base rate). ROC-AUC incorporates False Positive Rate, which includes the enormous pool of True Negatives (94% healthy accounts) in its denominator. This causes ROC-AUC to appear unrealistically high (e.g. 0.98+) even when precision on churners is poor. PR-AUC focuses exclusively on the minority class. By optimizing for PR-AUC (0.8837) and calibrating our decision threshold to 0.81, we achieved 87.1% precision, ensuring Customer Success managers are not flooded with false alarms."*

### Q2: What kind of data drift did you monitor and how did you pick statistical thresholds?
> *"We separated telemetry monitoring into two statistical regimes: continuous behavioral features (e.g. 30-day login decay, seat utilization) and discrete contract structures (e.g. contract tier, duration). For continuous variables, we applied the Two-Sample Kolmogorov-Smirnov (KS) test with an alpha threshold of 0.05. For discrete tiers, we used the Population Stability Index (PSI) with a 0.10 warning threshold and 0.25 critical action threshold. When upstream product changes or outages alter usage distributions, our webhook fires before model performance drops in production."*

### Q3: How did you incorporate SHAP TreeExplainer into the serving path without breaching latency SLAs?
> *"SHAP KernelExplainer is computationally intractable for real-time serving. We utilized `shap.TreeExplainer`, which has $O(TLD^2)$ complexity. To eliminate per-request overhead, the explainer is instantiated during FastAPI server startup inside the lifespan context manager. By pre-allocating the explainer and computing attribution directly on the transformed array, end-to-end single-account inference including top-3 driver extraction executes in **15.57ms (P50)**, comfortably below our 35ms production SLA."*

### Q4: How did you ensure zero data leakage in your feature pipeline?
> *"We enforced strict featurization ordering. The raw telemetry dataset is split into stratified train and test partitions BEFORE any transformer is fitted. The `SaaSFeatureTransformer` computes imputation statistics (such as median NPS) and categorical mappings strictly on the training partition. The test split and real-time inference payloads are transformed using the fitted transformer parameters without touching target labels."*

---

## 9. Google XYZ Resume Bullets

* **Engineered an active B2B SaaS churn early-warning microservice with LightGBM and FastAPI in Docker, achieving an 88.4% PR-AUC and sub-16ms inference latency on 10k enterprise account records.**
* **Integrated SHAP TreeExplainer into the inference layer to dynamically extract the top 3 behavioral risk drivers (e.g. seat underutilization, ticket escalation velocity) alongside prescriptive Customer Success playbooks.**
* **Built an automated continuous drift monitoring suite with Evidently AI and GitHub Actions CI/CD, triggering automated Slack alerts upon detecting covariate shift (Two-sample KS-test p < 0.05, PSI > 0.10).**

---

## 10. License

Distributed under the MIT License. See `LICENSE` for more information.
