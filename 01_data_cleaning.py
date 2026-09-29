"""
01_data_cleaning.py
Cleans the Telco Customer Churn dataset (Kaggle) and creates derived features.

Input : data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Output: data/telco_churn_clean.csv

Run: python 01_data_cleaning.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_PATH = DATA_DIR / "telco_churn_clean.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 1. Fix TotalCharges dtype ---
    # TotalCharges is read as object because 11 rows contain a blank string
    # (all of them have tenure == 0, i.e. brand-new customers who haven't been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # --- 2. Handle missing values ---
    # Fill the resulting NaNs with 0 rather than dropping rows or imputing the mean,
    # because a 0 is the factually correct value for a customer with zero tenure.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- 3. Standardize the target variable ---
    # Convert Yes/No to 1/0 so it can be used directly by sklearn metrics and models later.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- 4. Fix SeniorCitizen dtype ---
    # It's stored as 0/1 integers but is logically categorical, so cast to a labelled category
    # to keep it consistent with the other Yes/No demographic columns during EDA/plotting.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- 5. Drop the identifier column ---
    # customerID carries no predictive or analytical signal and would just add noise
    # to groupbys and, later, to the model's feature matrix.
    df = df.drop(columns=["customerID"])

    # --- 6. Strip stray whitespace from object columns ---
    # A few Kaggle copies of this file have trailing spaces in category values
    # (e.g. "No internet service "), which would silently break groupby/value_counts.
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Derived feature 1: tenure_group ---
    # Raw tenure (0-72 months) is too granular for clean business storytelling;
    # bucketing it into lifecycle stages makes churn-by-cohort analysis readable.
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # --- Derived feature 2: num_services ---
    # Counts how many of the eight phone, internet, and streaming services
    # a customer subscribes to.
    # This single number is a strong, business-intuitive proxy for "wallet share"
    # / product stickiness that no single raw column captures on its own.
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # --- Derived feature 3: avg_monthly_spend_consistency ---
    # Ratio of actual TotalCharges to (tenure * MonthlyCharges). Values far below 1
    # flag customers who've had billing pauses/discounts/downgrades — a useful
    # engagement-health signal that's invisible in the raw columns.
    expected_total = df["tenure"].replace(0, 1) * df["MonthlyCharges"]
    df["avg_monthly_spend_consistency"] = (df["TotalCharges"] / expected_total).round(2)

    return df


def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = add_derived_features(df)

    print("\nMissing values after cleaning:\n", df.isna().sum().sum())
    print("\nDtypes:\n", df.dtypes)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()
