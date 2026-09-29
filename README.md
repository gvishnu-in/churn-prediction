# Customer Churn Analysis — Telco Dataset

## Problem Statement
Telecom companies lose 15–25% of their customer base to churn annually, and
acquiring a new customer costs 5–7x more than retaining an existing one. This
project analyzes the Telco Customer Churn dataset (Kaggle) to identify **who
is churning, why, and which levers the business can pull** to reduce it —
then builds a classification model to flag at-risk customers before they leave.

## Tools Used
- **Python (pandas, NumPy)** — data cleaning and feature engineering
- **scikit-learn** — Logistic Regression & Random Forest classification
- **Matplotlib/Seaborn** (optional, for local plots) or **Power BI / Tableau
  Public** — dashboard layer
- **Jupyter/VS Code** — development environment

## Repo Structure
```
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw Kaggle download
│   └── telco_churn_clean.csv                  # output of step 1
├── requirements.txt
├── 01_data_cleaning.py
├── 02_eda_analysis.py
├── 03_classification_model.py
└── README.md
```

## How to Run
The raw CSV is included in `data/`. If it is missing, download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into
that folder.

### Linux or WSL (recommended if Windows blocks Python packages)

Open Ubuntu (WSL) and go to the project folder. Replace `user` in the path
with your Windows account name if it is different:

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

Then run the three scripts with
`.\.venv\Scripts\python.exe 01_data_cleaning.py` (replace the filename with
`02_eda_analysis.py` or `03_classification_model.py` for the next steps).
Some Windows systems may block pandas' compiled files through Code Integrity;
if that happens, use the Linux/WSL instructions above rather than disabling
Windows security policy.

The cleaning step creates or replaces `data/telco_churn_clean.csv`. Run the
EDA and model steps after cleaning. The scripts resolve data paths relative
to their own files. Load the cleaned CSV into Power BI or Tableau if you want
to build a dashboard; no dashboard file is included in this project.

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
- Customers with 0 counted services churn at 43.8%, dropping to single digits
  for customers with 7–8 services — bundling increases stickiness.

## Model Performance
*(Example output from `03_classification_model.py` in the WSL environment,
80/20 train-test split, 7,043 customers. Results may vary with package versions.)*

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.801 | 0.659 | 0.516 | 0.845 |
| Random Forest | 0.752 | 0.522 | 0.794 | 0.844 |

Both models land around **0.84 ROC-AUC**. Logistic Regression is more
precise (fewer false alarms); Random Forest caught about **79%** of actual
churners in this run (versus 52% for Logistic Regression) — a useful option
for a proactive retention program where missing an at-risk customer is
costlier than a wasted outreach call.

**Top 3 predictive features (Random Forest):** `tenure`, `TotalCharges`,
and `Contract_Two year` — confirming that how long a customer has stayed,
and whether they're contractually locked in, dominate churn risk more than
any demographic factor.

## Business Recommendation
Churn at this company is concentrated in a predictable segment — new,
month-to-month, electronic-check customers on fiber plans with few add-on
services — which means it is both identifiable and preventable. I recommend
targeting this segment with a 90-day onboarding retention program (proactive
check-ins, an incentivized switch to annual contracts, and a nudge toward
autopay) paired with a lightweight churn-risk score embedded in the CRM to
flag at-risk accounts for the retention team before cancellation — turning
churn management from a reactive, post-hoc exercise into a targeted,
model-driven one.

## Dashboard Plan
The dashboard is a suggested next step and is not included in this project.
A useful executive dashboard could include a churn-rate KPI card, churn by
contract and tenure cohort, monthly charges by churn status, churn by payment
method, and a filterable churn-risk customer table.
