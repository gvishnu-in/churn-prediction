# Customer Churn Analysis — Telco Dataset

[![Tests and dashboard deployment](https://github.com/gvishnu-in/churn-prediction/actions/workflows/deploy-dashboard.yml/badge.svg)](https://github.com/gvishnu-in/churn-prediction/actions/workflows/deploy-dashboard.yml)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)

**Live dashboard:** [Telco Customer Churn Dashboard](https://gvishnu-in.github.io/churn-prediction/)

**Project guide:** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

## Problem Statement
This project analyzes the Telco Customer Churn dataset to identify customer
groups with different historical churn rates, compare classification models,
and provide a calibrated score for prioritizing retention outreach. The
observational results are hypotheses for experiments, not causal findings.

## Tools Used
- **Python (pandas, NumPy)** — data cleaning and feature engineering
- **scikit-learn** — Logistic Regression & Random Forest classification
- **Plotly** — interactive dashboard in a standalone HTML file (no Power BI
  installation required)
- **Jupyter/VS Code** — development environment

## Repo Structure
```
├── .github/
│   └── workflows/
│       └── deploy-dashboard.yml
├── .gitignore
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw Kaggle download
│   └── telco_churn_clean.csv                  # output of step 1
├── models/
│   ├── churn_model.joblib                     # calibrated scoring pipeline
│   └── model_metadata.json                    # training provenance and metrics
├── monitoring/
│   └── drift.py                               # PSI drift-monitoring stub
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model_artifact.py
│   ├── test_model_utils.py
│   ├── test_score_customers.py
│   └── test_drift.py
├── app.py                                      # FastAPI prediction service
├── Dockerfile
├── cost_assumptions.py
├── model_utils.py
├── requirements.txt
├── requirements-dev.txt
├── score_customers.py                          # ranked CSV batch scorer
├── 01_data_cleaning.py
├── 02_eda_analysis.py
├── 03_classification_model.py
├── 04_create_dashboard.py
├── docs/PROJECT_GUIDE.md
└── README.md
```

## How to Run
The raw CSV is included in `data/`. If it is missing, download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into
that folder.

### Linux or WSL (recommended if Windows blocks Python packages)

If you are at a Windows PowerShell prompt, first enter Ubuntu WSL:

```powershell
wsl.exe -d Ubuntu
```

Run the remaining Linux commands **inside the Ubuntu/WSL terminal**, not in
PowerShell. Replace `user` in the path with your Windows account name if it
is different:

```bash
cd /mnt/c/Users/user/Downloads/churn_project/churn_project
```

If Python's virtual-environment support is not installed, install it first:

```bash
sudo apt update
sudo apt install python3-venv
```

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv ~/churn-venv
source ~/churn-venv/bin/activate
python -m pip install -r requirements.txt
```

Run the scripts in order:

```bash
python 01_data_cleaning.py
python 02_eda_analysis.py
python 03_classification_model.py
python 04_create_dashboard.py
```

When you run the project again later, open Ubuntu, return to the project
folder, and activate the existing environment with
`source ~/churn-venv/bin/activate` before running the scripts.

### Windows

From PowerShell in this project folder, create a virtual environment and
install dependencies with:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the four scripts from the project folder in this order:

```powershell
.\.venv\Scripts\python.exe 01_data_cleaning.py
.\.venv\Scripts\python.exe 02_eda_analysis.py
.\.venv\Scripts\python.exe 03_classification_model.py
.\.venv\Scripts\python.exe 04_create_dashboard.py
```

The dashboard is written to `dashboard/churn_dashboard.html`.
Some Windows systems may block pandas' compiled files through Code Integrity;
if that happens, use the Linux/WSL instructions above rather than disabling
Windows security policy.

The cleaning step creates or replaces `data/telco_churn_clean.csv`. Run the
EDA, model, and dashboard steps after cleaning. The scripts resolve data
paths relative to their own files. The dashboard step creates
`dashboard/churn_dashboard.html` and `dashboard/index.html`, standalone
interactive pages that you can open in a web browser; Power BI or Tableau is
not required.

## Data notes

- The raw dataset has **7,043 rows and 21 columns**; the cleaned dataset has
  **7,043 rows and 23 columns**.
- The 11 blank raw `TotalCharges` values all belong to customers with
  `tenure = 0`. Cleaning converts them to numeric and sets them to `0`.
- Repeated feature rows are retained; the pipeline does not deduplicate
  customers.

## Key Findings
*(Computed by running `02_eda_analysis.py` on the full 7,043-row dataset.)*
- **Contract type has the largest observed churn-rate difference**: Month-to-month customers
  churn at **42.7%**, vs. 11.3% for one-year and just **2.8%** for two-year
  contracts.
- **Churn is front-loaded in the first year**: customers in their first
  0–6 months churn at **52.9%**, dropping to 9.5% by months 49–72.
- **Fiber optic customers churn more than DSL** — 41.9% vs. 19.0% — despite
  paying more, pointing to a price/value or service-quality gap.
- **Monthly charges differ by churn outcome**: churned customers pay a median $79.65/month
  vs. $64.43/month for retained customers.
- **Electronic check payers churn at 45.3%**, compared with automatic bank
  transfer (16.7%) or credit card (15.2%) —
  a billing-experience hypothesis to investigate.
- Customers with 0 counted services churn at 43.8%, compared with 5.3% for
  customers with 8 services; this is an association, not proof that bundling
  reduces churn.

## Model card

**Intended use:** prioritize customers for a retention review or offer test.
This is not an automated cancellation decision and is not a causal estimate.

**Data:** the source Telco CSV has 7,043 rows. Training reads the cleaned CSV
written by `01_data_cleaning.py`; the target is `Churn`, and `customerID` plus
EDA-only derived columns are excluded. The stratified 80/20 hold-out comparison uses random state 42.
Five-fold stratified cross-validation runs only on the training portion.
The final model is a calibrated Random Forest pipeline; its fit uses 5,634
training rows and 19 raw customer features.

### Training-split cross-validation

Mean +/- sample standard deviation across stratified folds. Recall is measured
at the largest recall with precision at least 0.60.

| Model | ROC-AUC | PR-AUC (average precision) | Recall at precision >= 0.60 |
|---|---:|---:|---:|
| Logistic Regression | 0.846 +/- 0.014 | 0.662 +/- 0.022 | 0.654 +/- 0.085 |
| Logistic Regression (balanced) | 0.846 +/- 0.014 | 0.660 +/- 0.022 | 0.654 +/- 0.090 |
| Random Forest (balanced) | 0.846 +/- 0.010 | 0.660 +/- 0.023 | 0.656 +/- 0.075 |

### Preserved hold-out comparison

These are the existing default class-label prediction results, retained so
they can be compared with the cross-validation results. The table uses the
same stratified hold-out split.

| Model | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.842 | 0.634 |
| Logistic Regression (balanced) | 0.738 | 0.504 | 0.783 | 0.842 | 0.633 |
| Random Forest (balanced) | 0.752 | 0.521 | 0.781 | 0.839 | 0.645 |

The original unweighted Logistic Regression and Random Forest scores match
the initial README values. The balanced Logistic Regression is an added
comparison. Similar ROC-AUC alone does not determine the operating model:
outreach capacity, precision/recall trade-offs, calibration, and offer costs
matter.

### Calibration and threshold

Isotonic calibration had a slightly lower mean validation-fold Brier score
(0.1354 +/- 0.0043, versus sigmoid 0.1354 +/- 0.0051) and was selected.
On the hold-out split, Brier score changed from 0.1634 before calibration
to 0.1377 after calibration. Ten-bin reliability summaries are printed by
the training script and stored in model metadata.

Risk bands are derived from the one-third and two-thirds quantiles of
out-of-fold calibrated training probabilities. The current saved cutoffs are
0.084730 and 0.327105; the API loads them from metadata rather than embedding
band constants in application code. Calibration and observed churn rates for
each band are stored in the model metadata; these rates are empirical
validation estimates, not guaranteed future frequencies.
On out-of-fold training predictions, the observed churn rates were 3.4%,
19.4%, and 56.9% in the low, medium, and high bands, respectively.

The cost selector uses explicitly illustrative assumptions: offer cost
$30, customer value $500, and save rate 0.30. The break-even probability is
`offer_cost / (save_rate * customer_value)` = 0.200. The validation-selected
threshold was 0.200624; it flagged 667 hold-out customers and caught 318
hold-out churners. The estimated validation net value is $112,464.01 under
these assumptions, not observed revenue or a campaign result. Override the
assumptions with CLI flags or `config/cost_assumptions.json`.

The fitted calibrated pipeline is saved to `models/churn_model.joblib`;
`models/model_metadata.json` records its version, training environment,
feature list, thresholds, assumptions, and evaluation results. Feature
importances from the uncalibrated Random Forest fit were
`Contract_Month-to-month` (0.1281), `tenure` (0.1230), and `TotalCharges`
(0.0891); they are predictive
associations, not causal effects. The batch scorer exports each customer's
three largest isolated feature-replacement probability deltas as JSON in
`top_3_drivers`; these are model explanations against training medians/modes,
not causal effects.

**Limitations:** the evaluation is a random hold-out from one cross-sectional
dataset, not a future-period test. Offer economics are assumptions, and
unseen categories are counted and mapped to an all-zero one-hot vector rather
than being treated as any known category. See the
[project guide](docs/PROJECT_GUIDE.md) for serving and monitoring caveats.

## Sources and claim provenance

- **Dataset:** [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Descriptive rates and dashboard figures:** recomputable from the included cleaned data with [`01_data_cleaning.py`](01_data_cleaning.py), [`02_eda_analysis.py`](02_eda_analysis.py), and [`04_create_dashboard.py`](04_create_dashboard.py). These are observational associations, not causal findings.
- **Model metrics and calibration summaries:** generated by [`03_classification_model.py`](03_classification_model.py) and retained in [`models/model_metadata.json`](models/model_metadata.json). The split, random seed, CV strategy, and assumptions are documented above.
- **Metric definitions:** [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) and [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- **Campaign economics:** the offer cost, customer value, and save rate are explicitly illustrative defaults in [`cost_assumptions.py`](cost_assumptions.py), not measured outcomes.

## Recommendations

- **Convert month-to-month customers to annual plans:** month-to-month churn
  is **42.7%**, compared with **11.3%** for one-year contracts. Test a
  targeted annual-plan incentive and measure conversion, retention, and
  discount cost.
- **Strengthen first-six-month onboarding:** churn is **52.9%** in the
  first six months. Add proactive check-ins and service/setup support during
  this high-risk period.
- **Reduce electronic-check friction:** electronic-check churn is **45.3%**,
  versus **15.2–16.7%** for automatic credit-card and bank-transfer payments.
  Encourage autopay and investigate billing issues before cancellation.
- **Investigate fiber service value and quality:** fiber-optic churn is
  **41.9%**, compared with **19.0%** for DSL. Review pricing, reliability,
  and support for fiber customers before choosing a retention offer.
- **Prioritize offers using the model:** use the Random Forest probability to
  prioritize outreach, and monitor offer acceptance and retention outcomes.
  The score is a prioritization signal, not proof that any one factor causes
  churn.

## Dashboard
The generated dashboard includes overall churn KPIs and interactive charts
for churn by contract, tenure, internet service, payment method, and number
of services, plus monthly charge distributions for retained and churned
customers. Hover for chart details; the Plotly toolbar supports zooming,
panning, and downloading chart images. It is a local static HTML report and
does not need a server or Power BI. It summarizes historical patterns and
does not display individual churn predictions.

## Run the prediction API locally

When Windows blocks pandas, run these commands in order. First enter WSL
from PowerShell:

```powershell
wsl.exe -d Ubuntu
```

At the Ubuntu/WSL prompt, run:

```bash
cd /mnt/c/Users/user/Downloads/churn_project/churn_project
source ~/churn-venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn app:app --reload
```

The `source` command and Linux `/mnt/c/...` path only work inside WSL; do
not enter them at a `PS ...>` PowerShell prompt. `python -m uvicorn` runs
Uvicorn through the active WSL Python environment, so it does not depend on
the shell finding a separate `uvicorn` command.

Interactive API documentation is at <http://127.0.0.1:8000/docs>. `GET /health`
reports the loaded model version. Send every raw customer feature
(excluding `customerID` and `Churn`) to `POST /predict`; the response includes
churn probability, a risk band derived from model metadata, and `model_version`.
The default tenure validation ceiling is 120 months and can be overridden
with `TENURE_UPPER_BOUND`; it is intentionally above the training data's
observed maximum, so such long-tenure scores are extrapolations. The batch
route `POST /predict/batch` accepts a JSON array of up to 1,000 validated
customers and returns one prediction per row.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.7,
    "TotalCharges": 85.7
  }'
```

Run the tests after installing development dependencies with
`python -m pip install -r requirements-dev.txt`, then
`python -m pytest`.

## Deployment

### GitHub Pages dashboard

The GitHub Actions workflow runs the tests, regenerates the dashboard, and
deploys it when code is pushed to `main` (or started manually). A separate CI
job builds the API Docker image and checks `/health`; there is no automated
API deployment. Before publishing:

1. In the repository, open **Settings → Pages** and set **Build and
   deployment → Source** to **GitHub Actions**.
2. Push the project to the `main` branch.
3. Check the **Actions** tab and wait for **Deploy dashboard** to finish.
   Deployment is blocked if tests fail.
4. Open the live dashboard at
   <https://gvishnu-in.github.io/churn-prediction/>. The workflow deploys
   `dashboard/index.html` at this root URL.

The workflow generates both dashboard HTML files from the checked-in cleaned
CSV. The generated HTML is ignored by Git and does not need to be committed.

### Prediction API

Build and run the API container locally:

```powershell
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
curl.exe --fail http://127.0.0.1:8000/health
```

The image uses a non-root user and has a Docker healthcheck. The model and
metadata are loaded once at application startup. For production, connect the
repository to a Docker hosting service and deploy the image; that hosting
step is manual and is not part of the checked-in GitHub Actions workflow.
