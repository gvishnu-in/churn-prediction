# Customer Churn Analysis — Telco Dataset

**Live dashboard:** `https://<username>.github.io/<repo-name>/` (replace with the published URL)

**Screenshots:** _TODO: add dashboard screenshots to `docs/screenshots/`._

## Problem Statement
Telecom companies lose 15–25% of their customer base to churn annually, and
acquiring a new customer costs 5–7x more than retaining an existing one. This
project analyzes the Telco Customer Churn dataset (Kaggle) to identify **who
is churning, why, and which levers the business can pull** to reduce it —
then builds a classification model to flag at-risk customers before they leave.

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
│   └── churn_model.joblib                     # fitted prediction pipeline
├── tests/
│   ├── test_api.py
│   └── test_data.py
├── app.py                                      # FastAPI prediction service
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── 01_data_cleaning.py
├── 02_eda_analysis.py
├── 03_classification_model.py
├── 04_create_dashboard.py
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
- After dropping `customerID`, **22 feature rows are identical**. They are
  likely distinct customers with the same attributes, so they are kept
  intentionally rather than deduplicated.

## Key Findings
*(Computed by running `02_eda_analysis.py` on the full 7,043-row dataset.)*
- **Contract type is the strongest churn lever**: Month-to-month customers
  churn at **42.7%**, vs. 11.3% for one-year and just **2.8%** for two-year
  contracts.
- **Churn is front-loaded in the first year**: customers in their first
  0–6 months churn at **52.9%**, dropping to 9.5% by months 49–72.
- **Fiber optic customers churn more than DSL** — 41.9% vs. 19.0% — despite
  paying more, pointing to a price/value or service-quality gap.
- **Price sensitivity is real**: churned customers pay a median $79.65/month
  vs. $64.43/month for retained customers.
- **Electronic check payers churn at 45.3%**, more than 2.5x the rate of
  customers on automatic bank transfer (16.7%) or credit card (15.2%) —
  a fixable billing-experience problem.
- Customers with 0 counted services churn at 43.8%, dropping to 5.3% for
  customers with 8 services — bundling increases stickiness.

## Model Performance

Fresh output from `03_classification_model.py` using a stratified 80/20
train-test split (random state 42) on all 7,043 raw customer rows. The
preprocessing pipeline operates on the original customer fields, excluding
the customer identifier and churn target.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.660 | 0.561 | 0.842 |
| Random Forest | 0.754 | 0.524 | 0.802 | 0.843 |

**Model choice:** Logistic Regression has higher precision (**0.660** vs.
**0.524**), while Random Forest has much higher recall (**0.802** vs.
**0.561**) at a similar ROC-AUC (**0.843** vs. **0.842**). If the goal is to
catch more churners for retention offers, Random Forest is the defensible
choice; expect more false-positive outreach than with Logistic Regression.

**Top 3 Random Forest feature importances:** `tenure` (**0.1808**),
`Contract_Two year` (**0.1208**), and `TotalCharges` (**0.1174**). These
are model importances, not causal effects.

The fitted Random Forest preprocessing-and-model pipeline is saved as
`models/churn_model.joblib`. Its input is one row of raw customer fields;
preprocessing is part of the pipeline so training and API scoring use the
same transformations.

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

Interactive API documentation is at <http://127.0.0.1:8000/docs>. The health
check is `GET /health`. Send every raw customer feature (excluding
`customerID` and `Churn`) to `POST /predict`; the response contains the
churn probability and an illustrative risk band (`low` below 0.33, `medium`
from 0.33 to below 0.66, and `high` at or above 0.66).

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
deploys it when code is pushed to `main` (or started manually). Before
publishing:

1. In the repository, open **Settings → Pages** and set **Build and
   deployment → Source** to **GitHub Actions**.
2. Push the project to the `main` branch.
3. Check the **Actions** tab and wait for **Deploy dashboard** to finish.
   Deployment is blocked if tests fail.
4. Open the site root at
   `https://<username>.github.io/<repo-name>/`. The workflow deploys
   `dashboard/index.html` at this root URL.

The workflow generates both dashboard HTML files from the checked-in cleaned
CSV. The generated HTML is ignored by Git and does not need to be committed.

### Prediction API

Deploy the repository as a **Docker Web Service** on Render, or an equivalent
Docker service on Railway or Fly.io. Connect the service to this repository
and deploy using the included `Dockerfile`; the model is loaded once when the
FastAPI app starts. Free tiers may sleep when idle, so the first request
after inactivity can take longer. After deployment, verify `/health` and
open `/docs` on the service URL.
