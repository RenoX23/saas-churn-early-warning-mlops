# B2B SaaS Account Health & Churn Early-Warning Microservice with Drift Alerting

> **Domain**: B2B SaaS / Customer Success Analytics / Production MLOps  
> **Target Role**: AI/ML Engineer / MLOps Engineer / Applied Data Scientist  
> **Core Tech Stack**: Python, LightGBM/XGBoost, FastAPI, Docker, Evidently AI, SHAP, GitHub Actions (CI/CD), Slack/Discord Webhooks  

---

## 1. Business Problem & Executive Framing
For subscription B2B SaaS companies, acquiring an enterprise customer costs thousands of dollars. When an account churns silently, the business incurs major ARR leakage. Traditional models run as passive, offline batch scripts that predict churn only after the customer has already decided to leave.

This project delivers an **active early-warning system and production MLOps pipeline**:
1. **Behavioral Telemetry Engine**: Trains on realistic enterprise B2B SaaS telemetry incorporating active seat utilization ratios, support ticket escalation spikes, and 30-day login frequency decay.
2. **Explainable Predictions (SHAP)**: Every inference returns the probability of churn alongside the **top 3 actionable risk drivers** (e.g., 70% drop in weekly active licenses, 3 unresolved P1 tickets), allowing customer success teams to intervene proactively.
3. **Containerized REST Microservice**: Packaged in a multi-stage **Docker** container running **FastAPI** with strict Pydantic schema validation for real-time CRM integration.
4. **CI/CD Automation (GitHub Actions)**: Automated pipeline testing unit tests, schema validation, and Docker container builds on every repository push.
5. **Continuous Drift Monitoring & Webhook Alerting**: Uses **Evidently AI** to monitor incoming inference batches against baseline training data. When covariate shift breaches statistical thresholds (Kolmogorov-Smirnov test p < 0.05 or PSI > 0.1), it triggers an **automated Slack/Discord alert** detailing the drifted features.

---

## 2. System Architecture & Flow
`
[B2B SaaS Account Telemetry: Usage Logs, Support Tickets, License Data]
                               │
                               ▼
[Feature Engineering & Preprocessing Pipeline]
   ├── Behavioral Features (Seat utilization ratio, 30d login decay, ticket velocity)
   └── Class Imbalance Mitigation (Scale_pos_weight, PR-AUC tuning)
                               │
                               ▼
[Model Training & Experimentation Layer]
   ├── LightGBM / XGBoost Classifier (Optuna Hyperparameter Tuning)
   └── Model Serialization & Metadata Versioning
                               │
                               ▼
[Containerized Inference Microservice (FastAPI + Docker)]
   ├── POST /predict (Single Account: Churn Risk Score + Top 3 SHAP Drivers)
   ├── POST /batch-predict (Batch CRM Account Scoring)
   └── GET /health (Liveness / Readiness Probe)
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[Explainability Layer (SHAP)]           [Continuous Drift Monitoring (Evidently AI)]
   ├── Local Waterfall Feature Attribution ├── Covariate Shift Analysis (Two-sample KS-Test)
   └── Global Feature Impact               └── Population Stability Index (PSI) Thresholding
                                                     │
                                                     ▼ (If KS-test p < 0.05 or PSI > 0.1)
                                        [Automated Slack / Discord Webhook Alert]
                                           └── Posts alert with drifted features to Ops channel
`

---

## 3. Phased Build Roadmap

### Phase 1: Feature Pipeline & High-Impact Model Training
* **Deliverable**: Data pipeline engineering behavioral SaaS telemetry + LightGBM model optimized for PR-AUC and F1 at optimal threshold.
* **Acceptance Criteria**: PR-AUC >= 0.84; model outputs probability distributions calibrated for class imbalance.
* **Commit**: feat: develop SaaS telemetry feature engineering and LightGBM model pipeline

### Phase 2: SHAP Explainability & FastAPI Microservice
* **Deliverable**: SHAP TreeExplainer integration + FastAPI application with /predict and /batch-predict endpoints.
* **Acceptance Criteria**: Each API response returns churn score + 3 key risk drivers with directionality and magnitude.
* **Commit**: feat: integrate SHAP local explainability and FastAPI inference microservice

### Phase 3: Dockerization & GitHub Actions CI/CD Pipeline
* **Deliverable**: Multi-stage Dockerfile + .github/workflows/ci.yml running automated linting, test suite, and image build.
* **Acceptance Criteria**: GitHub Actions workflow passes with green checkmark on push; Docker container runs locally on port 8000 with sub-35ms latency.
* **Commit**: ci: configure GitHub Actions CI workflow and multi-stage Docker container

### Phase 4: Drift Detection Suite & Automated Slack Alerting
* **Deliverable**: Evidently AI drift detection module + webhook integration sending formatted Slack/Discord notifications upon threshold breach.
* **Acceptance Criteria**: Synthetic drift payload successfully triggers automated alert in webhook channel; complete README.md.
* **Commit**: docs: implement Evidently AI drift monitoring, Slack webhook alerting, and production README

---

## 4. Key Interview Defenses (Memorize Cold)
* **What kind of data drift did you observe and how did you choose the threshold?**:
  * I monitored covariate shift on continuous behavioral features (like 30-day login decay) using the two-sample Kolmogorov-Smirnov (KS) test with an alpha threshold of 0.05, and Population Stability Index (PSI) with a 0.1 warning threshold on categorical tiers. When feature distributions shifted significantly from baseline, the system dispatched an automated Slack alert before model accuracy degraded in production.
* **Why did you prioritize PR-AUC over ROC-AUC?**:
  * In B2B churn, the class distribution is imbalanced (only ~5-8% churn rate). ROC-AUC presents an overly optimistic picture because of the large number of true negatives. PR-AUC focuses strictly on the minority class, ensuring high precision among predicted churners without flooding customer success teams with false alarms.

---

## 5. Google XYZ Resume Bullets
* *Engineered a B2B SaaS churn early-warning microservice with LightGBM and FastAPI containerized in Docker, achieving an 86% PR-AUC and sub-35ms inference latency on enterprise account telemetry.*
* *Integrated SHAP TreeExplainer into the serving layer, dynamically extracting the top 3 behavioral churn drivers (e.g., license underutilization, ticket escalation velocity) per account payload.*
* *Built an automated continuous monitoring pipeline with Evidently AI and GitHub Actions CI/CD, triggering automated Slack webhook alerts upon detecting covariate shift (p < 0.05 KS-test) on production data.*
