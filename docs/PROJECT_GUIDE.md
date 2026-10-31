# Master Project Guide & Technical Dossier: Telco Customer Churn

This portfolio guide documents the checked-in pipeline, current model
evaluation, serving contract, and limits of what the evidence supports. It is
structured for Markdown-to-PDF export. For the quick-start view, see
[README.md](../README.md).

## 1. Executive Project Overview & Repository Skeleton

The project cleans Telco customer records, reports historical churn
aggregates, compares Logistic Regression and Random Forest variants, calibrates
the selected Random Forest score, and serves individual or batch predictions
through FastAPI. It also provides ranked-CSV scoring with counterfactual
feature deltas, a PSI drift-monitoring CLI, a static Plotly dashboard, and
GitHub Actions checks. The public dataset contains 7,043 records; the
stratified 80/20 split used 5,634 training rows and 1,409 holdout rows.

```text
Raw CSV (data/WA_Fn-UseC_-Telco-Customer-Churn.csv)
   |
   v
01_data_cleaning.py -> data/telco_churn_clean.csv
                         |             |              |
                         v             v              v
                02_eda_analysis.py  03_classification_model.py  04_create_dashboard.py
                                       |                         |
                                       | split -> stratified CV  +-> dashboard HTML
                                       | -> calibration -> metadata
                                       v
                               models/churn_model.joblib
                               models/model_metadata.json
                                       |
                   +-------------------+-------------------+
                   v                                       v
          app.py: /predict, /predict/batch       score_customers.py
                   |                                       |
                   +-> Docker / healthcheck      ranked CSV + drivers

Raw reference CSV -> monitoring/drift.py (PSI comparison)
```

`01_data_cleaning.py` reads the raw CSV and writes the cleaned CSV.
`02_eda_analysis.py`, `03_classification_model.py`, and
`04_create_dashboard.py` read the cleaned CSV. Training selects the 19 raw
customer fields required by the serving schema; the target and EDA-only
derived columns are excluded. `model_utils.prepare_customer_features` maps
cleaned `SeniorCitizen` labels back to the API's 0/1 representation and
normalizes raw CSV values in the same way at scoring time.

### Repository map

```text
.
├── .github/workflows/deploy-dashboard.yml
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_churn_clean.csv
├── dashboard/
│   ├── churn_dashboard.html   # generated interactive report
│   └── index.html             # GitHub Pages entry point
├── docs/PROJECT_GUIDE.md
├── models/
│   ├── churn_model.joblib
│   └── model_metadata.json
├── monitoring/drift.py
├── tests/
│   ├── fixtures/scoring_customers.csv
│   ├── test_api.py
│   ├── test_cost_assumptions.py
│   ├── test_data.py
│   ├── test_drift.py
│   ├── test_model_artifact.py
│   ├── test_model_utils.py
│   └── test_score_customers.py
├── 01_data_cleaning.py
├── 02_eda_analysis.py
├── 03_classification_model.py
├── 04_create_dashboard.py
├── app.py
├── cost_assumptions.py
├── model_utils.py
├── score_customers.py
├── Dockerfile
├── .dockerignore / .gitignore
├── requirements.txt / requirements-dev.txt
└── README.md
```

| File or directory | Importance | Purpose |
|---|---|---|
| `01_data_cleaning.py` | Crucial | Cleans raw values and builds EDA-only derived fields. |
| `02_eda_analysis.py` | Crucial | Prints the historical group comparisons. |
| `03_classification_model.py` | Crucial | Reads cleaned data; runs stratified CV and hold-out comparisons, calibration, threshold selection, and artifact writing. |
| `04_create_dashboard.py` | Crucial | Produces the static historical dashboard. |
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Crucial | Public source data and PSI reference. |
| `data/telco_churn_clean.csv` | Generated input | Feeds EDA, model training, and dashboard. |
| `models/churn_model.joblib` | Crucial | Calibrated model with preprocessing included. |
| `models/model_metadata.json` | Crucial | Model version, provenance, calibration, cutoffs, driver references, assumptions, and evaluation results. |
| `app.py`, `model_utils.py` | Crucial | Validated API, shared feature normalization, risk bands, OOV counting, and driver deltas. |
| `score_customers.py` | Useful | Produces ranked CSVs with per-row top-three model deltas and labeled lift/gains reports. |
| `monitoring/drift.py` | Useful stub | Calculates numeric and categorical PSI; it is not scheduled or deployed. |
| `tests/` | Crucial for verification | API, data-shape, artifact, OOV, scoring, cost-config, and PSI tests. |
| `.github/workflows/deploy-dashboard.yml` | Crucial for CI/deploy | Tests and publishes the dashboard; also builds and smoke-checks the API image. |
| `Dockerfile`, `.dockerignore` | Crucial for container | Defines the non-root API image and its healthcheck/build context. |
| `README.md`, `docs/PROJECT_GUIDE.md` | Reference | Quick start and detailed technical/interview guide. |
| `requirements.txt`, `requirements-dev.txt` | Crucial | Exact runtime and development dependency pins. |
| `.venv/`, `__pycache__/`, `.pytest_cache/` | Ignorable local state | Environment and generated caches; do not commit them. |
| `dashboard/` HTML outputs | Generated | Historical charts; no individual predictions are displayed. |

`tests/test_api.py` covers healthy startup, documentation access, valid
prediction shape/version, invalid tenure and category, batch scoring and
limits, and structured logs. It does not benchmark latency. `tests/test_data.py`
checks the cleaned CSV's shape/required columns, blank values, and binary
target; it does not execute the cleaning functions or validate model quality.

## 2. Theory, Mathematical Concepts & Tooling Breakdown

### Class imbalance and weights

The raw dataset's observed churn fraction is 0.2654 (26.54%). A classifier can
therefore obtain deceptively high accuracy by favoring the retained class.
For `class_weight="balanced"`, scikit-learn uses the class weight

```text
weight(class) = n_samples / (n_classes * n_samples_in_class)
```

The Random Forest uses this setting. The original Logistic Regression is
unweighted; a second Logistic Regression with balanced weights is included
for a fairer comparison. There is no oversampling, undersampling, or other
resampling. Class weighting changes the fitting loss and score distribution;
it does not change the true customer churn prior. Balanced-model scores must
not be interpreted as real-world churn frequencies without calibration on
representative data.

### Metrics and model comparison

- **Precision** is the fraction of flagged customers who actually churned;
  it helps estimate wasted outreach.
- **Recall** is the fraction of churners flagged; it measures how many
  potential churners the outreach list captures.
- **PR-AUC** is average precision and summarizes the precision/recall
  trade-off. It is especially informative when the positive class is the
  smaller class.
- **ROC-AUC** measures ranking discrimination across thresholds; it does not
  select an operating threshold or prove calibrated probabilities.
- **Accuracy** is correct classifications divided by all classifications and
  does not communicate the cost of false positives versus missed churners.

For the positive class `Churn = 1`, `precision = TP / (TP + FP)` and
`recall = TP / (TP + FN)`. ROC-AUC measures the probability that a randomly
selected churner ranks above a retained customer. PR-AUC is average precision
over the precision-recall curve. The observed churn prevalence is 26.54%, so
a no-skill PR baseline is approximately 0.265. Neither AUC selects a
deployment threshold.

The split is stratified 80/20 with `RANDOM_STATE = 42`. Stratified five-fold
CV runs only inside the training portion. Means and sample standard
deviations below are from the fresh `03_classification_model.py` run.
Recall-at-precision is the maximum recall at precision of at least 0.60.
Preprocessing lives inside each fold's estimator pipeline, so numeric
imputation/scaling and categorical imputation/encoding are fit on that
fold's training rows only. The holdout is reserved for final comparison and
calibration reporting, not fitting or selecting risk bands.

| Training-split CV model | ROC-AUC mean +/- SD | PR-AUC mean +/- SD | Recall at precision >= 0.60, mean +/- SD |
|---|---:|---:|---:|
| Logistic Regression | 0.846 +/- 0.014 | 0.662 +/- 0.022 | 0.654 +/- 0.085 |
| Logistic Regression (balanced) | 0.846 +/- 0.014 | 0.660 +/- 0.022 | 0.654 +/- 0.090 |
| Random Forest (balanced) | 0.846 +/- 0.010 | 0.660 +/- 0.023 | 0.656 +/- 0.075 |

The preserved hold-out comparison uses each estimator's default `predict`
classification threshold. It is not the cost-selected outreach threshold.

| Hold-out model | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.842 | 0.634 |
| Logistic Regression (balanced) | 0.738 | 0.504 | 0.783 | 0.842 | 0.633 |
| Random Forest (balanced) | 0.752 | 0.521 | 0.781 | 0.839 | 0.645 |

Logistic Regression is a simpler linear decision function; the Random Forest
can represent nonlinear splits and interactions. Their hold-out ROC-AUCs are
close, while their default-threshold precision/recall differ. Choose an
operating model based on outreach capacity, cost, and validated outcomes,
not on a small ROC-AUC difference alone. No hyperparameter search or
feature-selection procedure is implemented.

### Calibration, reliability, and threshold selection

`CalibratedClassifierCV` compares isotonic and sigmoid calibration using
stratified validation folds. Isotonic was selected with mean validation Brier
score 0.1354 +/- 0.0043 (sigmoid: 0.1354 +/- 0.0051). On the holdout,
Brier score improved from 0.1634 before calibration to 0.1377 after
calibration. Ten equal-width reliability bins before and after calibration
are stored in model metadata. Calibration improved the score but the
reliability curve is not perfect and can shift as the population changes.

The business inputs are explicitly **illustrative assumptions**, not
measured campaign results:

| Assumption | Current default | Rationale supplied for this project |
|---|---:|---|
| Offer cost | $30 per contacted customer | User-provided illustrative offer-cost assumption. |
| Customer value | $500 | Approximate $80 monthly charge based on the observed churner median of $79.65/month, with separately assumed margin and retained duration. |
| Save rate | 0.30 | Assumed fraction of would-be churners retained by an offer. |

The break-even probability is
`p* = offer_cost / (save_rate * customer_value)` = 0.200. The model script
chooses the candidate threshold maximizing

```text
expected net value = sum(flagged calibrated churn probabilities)
                     * save_rate * customer_value
                     - flagged customer count * offer_cost
```

Using out-of-fold calibrated training probabilities, the current selected
threshold is 0.200624. Under those assumptions it flagged 667 holdout
customers and caught 318 holdout churners; estimated validation net value
was $112,464.01. This is a model-based expectation under assumptions, not
realized revenue or incremental campaign impact.

The API risk bands are a separate display grouping: the lower and upper
cutoffs come from the one-third and two-thirds quantiles of out-of-fold
calibrated training probabilities. Current cutoffs are 0.084730 and 0.327105.
They are stored in artifact metadata, not fixed in API code. On these
out-of-fold rows, observed churn was 3.4% in low, 19.4% in medium, and 56.9%
in high risk; mean predicted probabilities were 3.1%, 19.3%, and 57.3%.
These are validation frequencies and not future guarantees. The cost
threshold should not be confused with these risk-band cutoffs.

### Tool stack and pinned dependencies

Package versions below are the versions reported by the Python environment
used for the fresh run. The Dockerfile uses the required Python 3.12 slim
base; the local training environment reported Python 3.11.2.

| Tool/package | Version |
|---|---:|
| Python (local training) | 3.11.2 |
| Python (container base) | 3.12-slim |
| pandas | 3.0.6 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| Plotly | 6.9.0 |
| joblib | 1.6.0 |
| FastAPI | 0.143.0 |
| Uvicorn | 0.54.0 |
| httpx (development) | 0.28.1 |
| pytest (development) | 8.4.2 |

## 3. Backend Serving (FastAPI) & Production Deployment

FastAPI's lifespan handler loads `churn_model.joblib` and
`model_metadata.json` once at startup. The calibrated classifier includes
the fitted preprocessing pipelines: numeric inputs use fold-fitted median
imputation and scaling; categorical inputs use most-frequent imputation and
one-hot encoding with `drop=None` and `handle_unknown="ignore"`. Unknown CSV
categories are counted and produce an all-zero vector for that feature
without aliasing a known category. A single serialized scoring object keeps
inference transformations aligned with training. This reduces
**training-serving skew** caused by separate preprocessing implementations.
It is distinct from **data leakage**, which occurs when validation/test or
future information influences fitting or model selection.

Pydantic validates required fields, numeric bounds, and categorical
`Literal` values. `TENURE_UPPER_BOUND` defaults to 120 and is configurable by
environment variable (`ge=0`, default `le=120`). The training data reaches
72 months, so predictions above that are out-of-range. Random Forest trees
use learned split thresholds rather than extending a linear trend; scores
remain bounded probabilities but are not evidence-backed forecasts beyond
the observed range. Invalid categories, missing fields, and null categories
are rejected by the API with HTTP 422.

`POST /predict/batch` accepts a JSON array of 1 through 1,000 validated
customers and returns one prediction per row. CSV scoring is less strict:
unknown and missing categorical values are counted as OOV; the fitted
imputer handles missing values and the encoder ignores unknown values.
Numeric missing values are imputed using training medians. `score_customers.py`
adds a JSON `top_3_drivers` cell containing the three largest absolute,
isolated probability changes when a feature is replaced with its training
median/mode. These are local model contrasts, not causal effects or SHAP
values. The API logs structured JSON containing request ID, latency, model
version, probability (or batch aggregate), risk level, status/method/path,
and categorical OOV counts; customer attributes and identifiers are not
logged. `GET /health` reports model readiness and version.

Metadata includes the semver model version, UTC training time, training Git
SHA, Python and library versions, training row count, feature list, chosen
threshold, calibrated risk cutoffs and per-band validation frequencies,
driver reference values, cost assumptions, and evaluation metrics. At
startup, the API warns if installed scikit-learn differs from the training
version.

The Docker image uses `python:3.12-slim`, installs exact runtime pins, runs
as a dedicated non-root `app` user, and declares a native healthcheck against
`/health`. GitHub Actions tests the app, builds the image, smoke-checks the
health endpoint, and deploys the static dashboard. The API image is not
automatically deployed to a hosting service.

### Enterprise production gaps and hardening priorities

- Replace the single random split with time-based validation on later customer
  periods; monitor calibration and ranking after deployment.
- Establish authentication, authorization, TLS termination, rate limits,
  request-size limits, and secrets management at the API gateway/host.
- Use a model registry and immutable artifact/version promotion process;
  enforce a tested dependency/runtime compatibility policy.
- Measure p50/p95/p99 latency, throughput, memory, and batch behavior on the
  target host; current request logs capture latency but are not a benchmark.
- Assess calibration and error rates across protected/customer groups, define
  privacy retention for logs, and obtain legal/business review before
  production outreach.
- Turn PSI/OOV reports into owned alerts with thresholds, labeled outcome
  monitoring, incident response, and a tested rollback/retraining procedure.
- Validate offer economics using randomized experiments. Current business
  values are assumptions, not outcomes or causal estimates.

## 4. Executive Business Analysis & Dashboard Insights

The following group differences are from `02_eda_analysis.py` on the full
dataset derived from the [Telco Customer Churn source data on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
They describe association, not the effect of an intervention. The generated
static dashboard is published at
[gvishnu-in.github.io/churn-prediction](https://gvishnu-in.github.io/churn-prediction/).

| Finding | Candidate action | KPI to test |
|---|---|---|
| Month-to-month churn is 42.7%, versus 11.3% for one-year and 2.8% for two-year contracts. | Randomize an annual-plan incentive among eligible month-to-month customers. | Incremental retained customers, conversion, discount cost, and net margin. |
| Churn is 52.9% in the 0-6 month tenure group and 9.5% in the 49-72 month group. | Test onboarding calls and setup support for new customers. | Retention at a prespecified follow-up, support contacts, and campaign cost. |
| Fiber-optic churn is 41.9%, versus 19.0% for DSL. | Investigate reliability, support, and price/value before choosing a service intervention. | Complaints, service incidents, retention, and margin by randomized treatment. |
| Electronic-check churn is 45.3%, versus 16.7% for automatic bank transfer and 15.2% for automatic credit card. | Test payment setup and billing-friction improvements. | Autopay adoption, failed payments, churn, and customer complaints. |
| Median monthly charge is $79.65 for churned customers and $64.43 for retained customers. | Test a targeted offer only if its margin economics are positive. | Incremental retention, discount exposure, and net contribution. |

The static dashboard built by `04_create_dashboard.py` presents overall
customer/churn/charge KPIs plus interactive churn-rate views for contract,
tenure, internet service, payment method, and service count, and a monthly
charge distribution by churn status. It is a historical aggregate report:
it does not render predictions or prove intervention impact. Use the
high-risk model segment to prioritize a controlled outreach experiment, not
to target every customer automatically.

For the default illustrative assumptions, the break-even rule is
`p * save_rate * customer_value - offer_cost > 0`, or
`p > $30 / (0.30 * $500) = 0.20`. At a 0.20 risk probability, the expected
saved-customer value is $30 and offsets the $30 offer cost; above it, the
assumed expected value is positive. This arithmetic excludes contact costs,
discount duration, margin, treatment effects, and customer eligibility, so
the current $112,464 validation estimate is not a forecast of realized
campaign benefit.

Randomly assign eligible customers to treatment and holdout before contact;
analyze outcomes by assignment (intention-to-treat) at a prespecified
follow-up, with guardrails for margin and complaints. Stratify or report
results by the pre-treatment risk groups and customer segments. This tests
whether an offer changes outcomes. Neither tree feature importance
(`Contract_Month-to-month` 0.1281, `tenure` 0.1230, `TotalCharges` 0.0891) nor
contract/payment associations establish causal effects.

## 5. Complete Command Reference Guide

Run these commands from the repository root in PowerShell. Uvicorn stays
running in its terminal; use a second terminal for the health check or API
requests.

```powershell
# Environment and dependencies
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# Sequential cleaning, analysis, training, and dashboard generation
.\.venv\Scripts\python.exe 01_data_cleaning.py
.\.venv\Scripts\python.exe 02_eda_analysis.py
.\.venv\Scripts\python.exe 03_classification_model.py
.\.venv\Scripts\python.exe 04_create_dashboard.py

# Tests
.\.venv\Scripts\python.exe -m pytest

# API server (leave this process running)
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
# In a second PowerShell terminal:
curl.exe --fail http://127.0.0.1:8000/health

# Ranked batch output, including top_3_drivers
.\.venv\Scripts\python.exe score_customers.py --input data\WA_Fn-UseC_-Telco-Customer-Churn.csv --output outputs\ranked_customers.csv

# Optional feature-drift report (PSI)
.\.venv\Scripts\python.exe monitoring\drift.py --input data\WA_Fn-UseC_-Telco-Customer-Churn.csv

# Docker API image and health endpoint
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
# In a second terminal:
curl.exe --fail http://127.0.0.1:8000/health

# Explicit Git staging, commit, push, and remote verification
git status --short
git add docs\PROJECT_GUIDE.md
git diff --cached --check
git commit -m "docs: publish master project guide and technical dossier"
git push origin main
git log origin/main --oneline -3
```

Training and scoring also accept `--offer-cost`, `--customer-value`,
`--save-rate`, and `--cost-config PATH`. Settings resolve from built-in
defaults, then an optional JSON file, then explicit CLI overrides. Invalid,
non-finite, or uneconomic values fail fast; the selected assumptions are
printed before training/scoring.

For Windows systems that block binary Python packages, the README also
documents using the same workflow inside Ubuntu/WSL. Do not mix PowerShell
commands with WSL-only `source` commands or `/mnt/c/...` paths.

### Known limitations and next steps

- Results use one public, cross-sectional dataset and a random stratified
  hold-out, not a time-based future-period evaluation.
- The cost, retained customer value, and save-rate inputs are assumptions.
  Replace them with measured campaign economics before using the threshold
  operationally.
- Calibration is improved by the reported Brier score but the hold-out
  reliability table is not perfectly aligned in every bin.
- `OneHotEncoder(drop=None, handle_unknown="ignore")` gives OOV categories an
  all-zero vector rather than aliasing a known category; OOVs are logged or
  printed but the model has no labeled examples of truly novel categories.
- The API validates its categorical values against known `Literal` choices;
  CSV scoring instead counts OOV values, imputes missing categories with the
  training mode, and relies on the encoder policy for unseen categories.
- The API accepts tenure through 120 months, but training spans only 0-72;
  larger-tenure scores are unsupported extrapolations even though tree
  predictions remain bounded.
- PSI at 0.2 is a review flag, not proof of performance degradation. The
  drift module is a command-line stub, has no alerting schedule, and does not
  retrain or evaluate labels automatically.
- No production latency benchmark, fairness assessment, campaign result,
  SHAP attribution, or automatic API deployment is included. Per-row driver
  deltas are isolated counterfactual contrasts against a fixed training
  reference, not SHAP values.
- Next steps: collect a later-period labeled sample, evaluate calibration and
  ranking over time, estimate offer economics with a randomized holdout, and
  define alert ownership and retraining criteria.

## 6. Technical Interview Preparation (Q&A)

### How do you handle class imbalance, and how would you choose between these models?

Churn prevalence is 26.54%. I compare unweighted Logistic Regression,
class-weight-balanced Logistic Regression, and a class-weight-balanced
Random Forest on the same stratified split and five folds; I do not resample.
Weights increase the minority class's influence during fitting but can shift
score calibration. The hold-out ROC-AUC is 0.842 for Logistic Regression and
0.839 for the Random Forest; default-threshold precision/recall are
0.657/0.559 and 0.521/0.781, respectively. I choose using validated
outreach capacity, calibration, and expected value—not a tiny AUC
difference—and recalibrate probabilities on representative data.

### What happens to an unseen category or a missing value?

The encoder retains all fitted levels (`drop=None`) and ignores unknown
levels, which become all-zero vectors instead of aliasing a known category.
The API rejects unknown `Literal` values, absent required fields, and null
categoricals with HTTP 422. CSV scoring counts these as OOV; missing
categoricals are imputed with the fitted training mode and numeric missing
values with training medians. OOV handling avoids a crash but cannot recover
information for a category absent from training.

### How do you prevent training-serving skew, and how is that different from leakage?

Training and CSV serving call the shared feature normalizer, and the fitted
imputers, scaler, encoder, and classifier are serialized together inside the
calibrated estimator. The API loads that artifact rather than reimplementing
preprocessing. Leakage is distinct: it is using validation/test or future
information during fitting. The stratified holdout is reserved for final
reporting, while fold-local pipelines prevent imputer/encoder fitting on
validation rows.

### How would you optimize API latency?

There is no production latency baseline, so I would not claim a speedup. The
API loads the model once during lifespan startup and accepts batches up to
1,000 rows. The CSV driver calculation batches its per-feature counterfactual
predictions to avoid a model call per feature. Request logs collect
`latency_ms`; next I would benchmark p50/p95/p99 for single and batch traffic,
profile preprocessing and prediction, then optimize based on measured
bottlenecks and a quality-preserving regression test.

### Why report metrics beyond accuracy, and what does calibration add?

At 26.54% prevalence, accuracy can hide missed churn and wasted contacts.
ROC-AUC summarizes ranking over false-positive rates; PR-AUC emphasizes the
positive-class precision/recall trade-off. Calibration addresses a different
question: whether scores correspond to empirical frequencies. Isotonic
calibration reduced holdout Brier score from 0.1634 to 0.1377, though the
reliability bins still have error and require monitoring.

### How would you monitor drift and decide when to retrain?

The CLI compares scoring data with the raw training reference and reports
numeric and categorical PSI; PSI > 0.2 is a review flag, not proof of
degradation. The API and CSV scorer count OOV categories. Neither is an
automated retraining trigger. I would add later labeled performance and
calibration checks, alert ownership, time-based validation, and a tested
rollback/deployment comparison before retraining.

### What are the pitfalls of feature importance and contract effects?

The Random Forest's top impurity importances are
`Contract_Month-to-month` (0.1281), `tenure` (0.1230), and `TotalCharges`
(0.0891). Importance can be biased by correlated features and is not an
individual explanation. The CSV's top-three driver list instead reports
isolated probability deltas against reference values; neither is causal.
Contract churn-rate differences are observational; randomized holdout
experiments are needed to estimate intervention effects.

## Sources and claim provenance

- **Dataset and population description:** [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The repository includes its source CSV and a reproducible cleaned output.
- **Descriptive rates and dashboard figures:** generated from the local data by [`01_data_cleaning.py`](../01_data_cleaning.py), [`02_eda_analysis.py`](../02_eda_analysis.py), and [`04_create_dashboard.py`](../04_create_dashboard.py); these are descriptive associations, not causal estimates.
- **CV, holdout, calibration, risk-band, and feature-importance values:** produced by [`03_classification_model.py`](../03_classification_model.py) and recorded in [`models/model_metadata.json`](../models/model_metadata.json). The split, seed, metrics, and illustrative cost assumptions are described in this guide.
- **Class weighting and metric definitions:** [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), and [class-weight behavior](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).
- **Serving behavior:** [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/) and [Pydantic field constraints](https://docs.pydantic.dev/latest/concepts/fields/).
- **Campaign economics:** values are defaults in [`cost_assumptions.py`](../cost_assumptions.py), explicitly illustrative rather than measured customer or campaign outcomes.
