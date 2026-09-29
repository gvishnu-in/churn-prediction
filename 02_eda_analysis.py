"""
02_eda_analysis.py
Five business-relevant insights from the cleaned Telco churn dataset,
each with the exact pandas code that produces it.

Run: python 02_eda_analysis.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "telco_churn_clean.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("INSIGHT 1: Churn rate by contract type")
print("Why it matters: reveals whether month-to-month customers are the")
print("main churn driver -> directly informs contract-incentive strategy.")
print("=" * 70)
insight_1 = (
    df.groupby("Contract")["Churn"]
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename("churn_rate_%")
)
print(insight_1, "\n")


print("=" * 70)
print("INSIGHT 2: Churn rate by tenure group (customer lifecycle stage)")
print("Why it matters: shows exactly which lifecycle window is most at risk,")
print("so retention campaigns/onboarding fixes can be timed correctly.")
print("=" * 70)
insight_2 = (
    df.groupby("tenure_group")["Churn"]
    .mean()
    .mul(100)
    .round(1)
    .rename("churn_rate_%")
)
print(insight_2, "\n")


print("=" * 70)
print("INSIGHT 3: Churn rate by internet service type")
print("Why it matters: flags whether a specific service line (e.g. Fiber optic)")
print("has a quality/pricing problem driving disproportionate churn.")
print("=" * 70)
insight_3 = (
    df.groupby("InternetService")["Churn"]
    .agg(customers="count", churn_rate=lambda x: round(x.mean() * 100, 1))
    .sort_values("churn_rate", ascending=False)
)
print(insight_3, "\n")


print("=" * 70)
print("INSIGHT 4: Monthly spend gap between churned and retained customers")
print("Why it matters: quantifies whether churn is being driven by price")
print("sensitivity -> tells finance/product if a pricing tier fix is warranted.")
print("=" * 70)
insight_4 = (
    df.groupby("Churn")["MonthlyCharges"]
    .agg(avg_monthly_charge="mean", median_monthly_charge="median")
    .round(2)
    .rename(index={0: "Retained", 1: "Churned"})
)
print(insight_4, "\n")


print("=" * 70)
print("INSIGHT 5: Churn rate by payment method")
print("Why it matters: highlights friction in the billing experience")
print("(e.g. manual payment methods) that ops/billing teams can act on directly.")
print("=" * 70)
insight_5 = (
    df.groupby("PaymentMethod")["Churn"]
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename("churn_rate_%")
)
print(insight_5, "\n")


print("=" * 70)
print("BONUS: num_services vs churn (validates the derived feature)")
print("=" * 70)
bonus = (
    df.groupby("num_services")["Churn"]
    .mean()
    .mul(100)
    .round(1)
    .rename("churn_rate_%")
)
print(bonus)
