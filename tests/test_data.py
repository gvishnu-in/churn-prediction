from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "telco_churn_clean.csv"
EXPECTED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "tenure_group",
    "num_services",
    "avg_monthly_spend_consistency",
}


def test_cleaned_dataset_has_expected_shape_and_columns() -> None:
    data = pd.read_csv(DATA_PATH)

    assert len(data) == 7043
    assert EXPECTED_COLUMNS.issubset(data.columns)


def test_cleaned_dataset_has_no_blank_cells() -> None:
    data = pd.read_csv(DATA_PATH, dtype=str)

    assert not data.isna().any().any()
    assert not data.apply(
        lambda column: column.str.strip().eq("").any()
    ).any()


def test_churn_contains_only_binary_values() -> None:
    data = pd.read_csv(DATA_PATH)

    assert set(data["Churn"].unique()).issubset({0, 1})
