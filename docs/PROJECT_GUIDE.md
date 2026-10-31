# Project Guide: Telco Customer Churn

This guide describes the checked-in pipeline, the model trained in the current
environment, and the limits of what the observed results support. For the
short project overview, see [README.md](../README.md).

## Executive overview and data flow

The project cleans Telco customer records, reports historical churn
aggregates, compares three classifiers, calibrates a Random Forest score, and
serves customer or batch predictions through FastAPI. It also provides
ranked-CSV scoring, a PSI drift-monitoring stub, a static dashboard, and CI
checks. The dataset contains 7,043 records; the model-training run used 5,634
rows and held out 1,409 rows.

```text
Raw CSV (data/WA_Fn-UseC_-Telco-Customer-Churn.csv)
   |                         |
   | 01_data_cleaning.py     | 03_classification_model.py
   v                         |  stratified train/hold-out split
Clean CSV                    |  CV -> calibration -> metadata
   |            |            v
   |            |        churn_model.joblib + model_metadata.json
   |            |                    |
   |            |                    +------> app.py: /predict, /predict/batch
   |            |                    +------> score_customers.py
   |            |                    +------> monitoring/drift.py (raw reference)
   |            v
   +--------> 02_eda_analysis.py
   +--------> 04_create_dashboard.py -> dashboard HTML
```

`01_data_cleaning.py` reads the raw CSV and writes the cleaned CSV.
`02_eda_analysis.py` and `04_create_dashboard.py` read the cleaned CSV.
`03_classification_model.py` reads the raw CSV directly; the cleaned CSV is
not its input.

## Repository map

| File or directory | Importance | Purpose |
|---|---|---|
| `01_data_cleaning.py` | Crucial | Cleans raw values and builds EDA-only derived fields. |
| `02_eda_analysis.py` | Crucial | Prints the historical group comparisons. |
| `03_classification_model.py` | Crucial | Runs CV and hold-out comparisons, calibration, threshold selection, and artifact writing. |
| `04_create_dashboard.py` | Crucial | Produces the static historical dashboard. |
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Crucial | Raw reference and model-training source. |
| `data/telco_churn_clean.csv` | Generated input | Feeds EDA and dashboard, not the classifier. |
| `models/churn_model.joblib` | Crucial | Calibrated model with preprocessing included. |
| `models/model_metadata.json` | Crucial | Model version, provenance, cutoffs, assumptions, and evaluation results. |
| `app.py`, `model_utils.py` | Crucial | Validated API, logging, risk bands, and OOV counting. |
| `score_customers.py` | Useful | Produces ranked CSVs and labeled lift/gains reports. |
| `monitoring/drift.py` | Useful stub | Calculates numeric and categorical PSI; it is not scheduled or deployed. |
| `tests/` | Crucial for verification | API, data-shape, artifact, OOV, scoring, cost-config, and PSI tests. |
| `.github/workflows/deploy-dashboard.yml` | Crucial for CI/deploy | Tests and publishes the dashboard; also builds and smoke-checks the API image. |
| `Dockerfile`, `.dockerignore` | Crucial for container | Defines the non-root API image and its healthcheck/build context. |
| `README.md`, `docs/PROJECT_GUIDE.md` | Reference | Quick start and detailed technical/interview guide. |
| `.venv/`, `__pycache__/`, `.pytest_cache/` | Ignorable local state | Environment and generated caches; do not commit them. |
| `dashboard/` HTML outputs | Generated | Historical charts; no individual predictions are displayed. |

`tests/test_api.py` covers healthy startup, documentation access, valid
prediction shape/version, invalid tenure and category, batch scoring and
limits, and structured logs. It does not benchmark latency. `tests/test_data.py`
checks the cleaned CSV's shape/required columns, blank values, and binary
target; it does not execute the cleaning functions or validate model quality.

## ML theory as applied here

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
resampling.

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

The split is stratified 80/20 with `RANDOM_STATE = 42`. Stratified five-fold
CV runs only inside the training portion. Means and sample standard
deviations below are from the fresh `03_classification_model.py` run.
Recall-at-precision is the maximum recall at precision of at least 0.60.

| Training-split CV model | ROC-AUC mean +/- SD | PR-AUC mean +/- SD | Recall at precision >= 0.60, mean +/- SD |
|---|---:|---:|---:|
| Logistic Regression | 0.846 +/- 0.014 | 0.661 +/- 0.022 | 0.656 +/- 0.085 |
| Logistic Regression (balanced) | 0.846 +/- 0.014 | 0.660 +/- 0.022 | 0.652 +/- 0.086 |
| Random Forest (balanced) | 0.847 +/- 0.012 | 0.661 +/- 0.020 | 0.652 +/- 0.079 |

The preserved hold-out comparison uses each estimator's default `predict`
classification threshold. It is not the cost-selected outreach threshold.

| Hold-out model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.807 | 0.660 | 0.561 | 0.842 |
| Logistic Regression (balanced) | 0.739 | 0.505 | 0.783 | 0.842 |
| Random Forest (balanced) | 0.754 | 0.524 | 0.802 | 0.843 |

Logistic Regression is a simpler linear decision function; the Random Forest
can represent nonlinear splits and interactions. Their hold-out ROC-AUCs are
close, while their default-threshold precision/recall differ. Choose an
operating model based on outreach capacity, cost, and validated outcomes,
not on a small ROC-AUC difference alone. No hyperparameter search or
feature-selection procedure is implemented.

### Calibration, reliability, and threshold selection

`CalibratedClassifierCV` compares isotonic and sigmoid calibration using
stratified validation folds. Sigmoid was selected: its mean validation Brier
score was 0.1351 +/- 0.0060, compared with isotonic's 0.1352 +/- 0.0055. On
the hold-out data, Brier score fell from 0.1624 before calibration to 0.1368
after calibration. The script prints and metadata stores ten equal-width
reliability bins before and after calibration. Two of the upper-probability
bins are empty after calibration on this hold-out set; calibration does not
make every bin perfectly reliable.

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
threshold is 0.200090. Under those assumptions it flagged 651 hold-out
customers and caught 318 hold-out churners; expected validation net value
was $112,353.78. That figure is a model-based expectation under assumptions,
not realized revenue.

The API risk bands are a separate display grouping: the lower and upper
cutoffs come from the one-third and two-thirds quantiles of out-of-fold
calibrated training probabilities. Current cutoffs are 0.074306 and 0.344725.
They are stored in artifact metadata, not fixed in API code. The cost
threshold should not be confused with these risk-band cutoffs.

## Tooling and exact commands

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

From PowerShell in the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe 01_data_cleaning.py
.\.venv\Scripts\python.exe 02_eda_analysis.py
.\.venv\Scripts\python.exe 03_classification_model.py
.\.venv\Scripts\python.exe 04_create_dashboard.py
.\.venv\Scripts\python.exe -m pytest
```

The training run accepts `--offer-cost`, `--customer-value`, `--save-rate`,
and `--cost-config PATH`. Settings resolve per value in this order: defaults,
then the optional JSON file at `config/cost_assumptions.json`, then explicitly
provided flags. A missing config file uses defaults silently. Invalid,
non-finite, or uneconomic inputs fail fast. The current defaults and
break-even threshold are printed at the start of training.

Start the API locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
curl.exe --fail http://127.0.0.1:8000/health
```

Rank a raw customer CSV and compare feature drift:

```powershell
.\.venv\Scripts\python.exe score_customers.py --input data\WA_Fn-UseC_-Telco-Customer-Churn.csv --output outputs\ranked_customers.csv
.\.venv\Scripts\python.exe monitoring\drift.py --input data\WA_Fn-UseC_-Telco-Customer-Churn.csv
```

`score_customers.py` also accepts the cost-assumption flags and
`--cost-config PATH`. Its output includes `customer_id` if the input has an
identifier, churn probability, risk band, rank, and decile. Per-row
`top_3_drivers` are omitted: no SHAP dependency or permutation-based
explanations are implemented. Top-decile lift and cumulative gains are
printed only when the input includes valid `Churn` outcomes.

Build and run the Docker API:

```powershell
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
curl.exe --fail http://127.0.0.1:8000/health
```

The existing GitHub Actions workflow runs tests, regenerates the static
dashboard, and deploys GitHub Pages on a push to `main` or manual dispatch.
The added API job builds the Docker image and curls `/health`, checking that
the response includes `model_version`. API deployment itself is not automated
by this repository.

Trigger the dashboard deployment by pushing to `main`:

```powershell
git push origin main
```

Alternatively, start the `Deploy dashboard` workflow manually from GitHub
Actions. The API container deployment is a separate manual hosting action.

## Serving design

FastAPI's lifespan handler loads `churn_model.joblib` and
`model_metadata.json` once at startup. The calibrated classifier includes
the fitted preprocessing pipelines: numeric inputs are imputed and scaled;
categorical inputs are imputed with a missing marker then one-hot encoded
with `drop="first"` and `handle_unknown="ignore"`. A single serialized
scoring object ensures that inference uses the same fitted transformations
as training. This prevents **training-serving skew** caused by separately
implemented preprocessing. It is distinct from **data leakage**, which is
when information unavailable at prediction time (or from validation/test
data) influences training.

Pydantic validates required fields, numeric bounds, and categorical
`Literal` values. `TENURE_UPPER_BOUND` defaults to 120 and is configurable by
environment variable. The raw training data reaches 72 months; allowing a
larger bound supports longer-tenure records but does not add training
evidence for extrapolation. Invalid category values, missing fields, and
null categorical values are rejected by the API with HTTP 422.

`POST /predict/batch` accepts a JSON array of 1 through 1,000 validated
customers and returns one prediction per row. CSV scoring is less strict:
unknown and missing categorical values are counted as OOV and the fitted
encoder maps unknown transformed values to all zeros; numeric missing values
are imputed. The API logs structured JSON containing a generated request ID,
latency, model version, probability (or batch aggregate), risk level,
status/method/path, and categorical OOV counts. It does not log customer
fields or identifiers. `GET /health` reports the loaded model version.

Metadata includes the semver model version, UTC training time, training Git
SHA, Python and library versions, training row count, feature list, chosen
threshold, risk cutoffs, cost assumptions, and evaluation metrics. At startup,
the API emits a warning if its installed scikit-learn version differs from
the recorded training version.

## Business translation and experiment design

The following group differences are from `02_eda_analysis.py` on the full
dataset. They describe association, not the effect of an intervention.

| Finding | Candidate action | KPI to test |
|---|---|---|
| Month-to-month churn is 42.7%, versus 11.3% for one-year and 2.8% for two-year contracts. | Randomize an annual-plan incentive among eligible month-to-month customers. | Incremental retained customers, conversion, discount cost, and net margin. |
| Churn is 52.9% in the 0-6 month tenure group and 9.5% in the 49-72 month group. | Test onboarding calls and setup support for new customers. | Retention at a prespecified follow-up, support contacts, and campaign cost. |
| Fiber-optic churn is 41.9%, versus 19.0% for DSL. | Investigate reliability, support, and price/value before choosing a service intervention. | Complaints, service incidents, retention, and margin by randomized treatment. |
| Electronic-check churn is 45.3%, versus 16.7% for automatic bank transfer and 15.2% for automatic credit card. | Test payment setup and billing-friction improvements. | Autopay adoption, failed payments, churn, and customer complaints. |
| Median monthly charge is $79.65 for churned customers and $64.43 for retained customers. | Test a targeted offer only if its margin economics are positive. | Incremental retention, discount exposure, and net contribution. |

Randomly assign eligible customers to treatment and holdout before contact;
analyze outcomes by assignment (intention-to-treat) at a prespecified
follow-up, with guardrails for margin and complaints. Stratify or report
results by the pre-treatment risk groups and customer segments. This tests
whether an offer changes outcomes. Neither tree feature importance
(`tenure` 0.1808, `Contract_Two year` 0.1208, `TotalCharges` 0.1174) nor
contract/payment associations establish causal effects.

## Known limitations and next steps

- Results use one public, cross-sectional dataset and a random stratified
  hold-out, not a time-based future-period evaluation.
- The cost, retained customer value, and save-rate inputs are assumptions.
  Replace them with measured campaign economics before using the threshold
  operationally.
- Calibration is improved by the reported Brier score but the hold-out
  reliability table still has uneven bins, including empty upper bins.
- `drop="first"` plus `handle_unknown="ignore"` silently aliases an unseen
  category with that feature's dropped baseline in model space. The policy is
  tested and OOV counts are logged, but a future model could use `drop=None`
  if a measured evaluation supports it.
- The API validates its categorical values against known `Literal` choices;
  CSV scoring instead counts OOV values and relies on the encoder policy.
- PSI at 0.2 is a review flag, not proof of performance degradation. The
  drift module is a command-line stub, has no alerting schedule, and does not
  retrain or evaluate labels automatically.
- No production latency benchmark, fairness assessment, campaign result,
  per-row SHAP explanation, or automatic API deployment is included.
- Next steps: collect a later-period labeled sample, evaluate calibration and
  ranking over time, estimate offer economics with a randomized holdout, and
  define alert ownership and retraining criteria.

## Interview Q&A

### How would you choose between Logistic Regression and the Random Forest when ROC-AUC is nearly equal?

The hold-out ROC-AUC is 0.842 for Logistic Regression and 0.843 for Random
Forest; their CV means are 0.846 and 0.847. At the default threshold,
Logistic Regression has precision/recall 0.660/0.561 and Random Forest
0.524/0.802. Balanced Logistic Regression reaches recall 0.783 at precision
0.505. I would choose based on validated campaign capacity and expected
incremental value, not the small hold-out AUC difference. For this
repository, the Random Forest is the retained final model and is calibrated.

### What happens to an unseen category or a missing value?

I ran a minimal `OneHotEncoder(drop="first", handle_unknown="ignore")` test
with scikit-learn 1.9.1: both the first fitted category and an unseen category
transform to `[0.0]`. The saved pipeline test confirms the same behavior and
counts the unseen value as OOV. The API rejects values outside its
`Literal` choices, absent required fields, and null categorical fields with
422. For CSV scoring, the categorical imputer supplies a missing marker;
because that marker was not in training vocabulary, the encoder treats it as
unknown and maps it to all zeros. Numeric missing values are imputed.

### How do you prevent training-serving skew, and how is that different from leakage?

The fitted preprocessing steps are stored inside the estimator pipeline
wrapped by `CalibratedClassifierCV`, and the API loads that artifact rather
than reimplementing scaling or category encoding. That keeps transformations
consistent. Leakage is a separate issue: it is using test/validation or
future information during fitting. CV and calibration are run within the
training portion, and the hold-out is used for final reporting.

### How would you optimize API latency?

There is no measured latency baseline in this repository, so I would not
claim a speedup. The API loads the model once at startup, and batch scoring
accepts up to 1,000 rows to avoid one network round trip per customer. Request
logs collect `latency_ms`; next I would benchmark representative single-row
and batch sizes, profile preprocessing versus prediction, and only then
consider model size, concurrency, or batching changes.

### Why report metrics beyond accuracy, and what does calibration add?

The observed churn fraction is 26.54%, so accuracy alone hides the
false-positive/missed-churn trade-off. The project reports ROC-AUC, PR-AUC,
precision, recall, and recall at precision >= 0.60. Calibration makes
probabilities more useful for expected-value decisions: hold-out Brier
changed from 0.1624 to 0.1368, though the ten-bin reliability summary is not
perfect.

### How would you monitor drift and decide when to retrain?

The CLI compares scoring data with the raw training CSV and reports numeric
and categorical PSI; PSI > 0.2 is a review flag. The API and batch scorer
also count OOV categories. Neither mechanism is an automated retraining
trigger. I would pair drift with later labeled performance and calibration,
define ownership/alert thresholds, and retrain only after time-based
validation and a deployment comparison.

### What are the pitfalls of feature importance and contract effects?

The Random Forest's top importances are `tenure` (0.1808),
`Contract_Two year` (0.1208), and `TotalCharges` (0.1174), but these are
predictive associations and can be affected by correlated features and the
importance method. Likewise, the 42.7% versus 2.8% contract churn-rate
difference is observational. A randomized offer holdout is needed to
estimate whether changing an offer or contract incentive changes retention.
