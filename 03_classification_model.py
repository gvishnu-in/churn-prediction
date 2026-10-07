"""
Compare churn classifiers, report feature importance, and persist the
Random Forest pipeline for scoring raw customer fields.

Run: python 03_classification_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"
TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def make_pipeline(
    customer_columns: list[str], model: BaseEstimator
) -> Pipeline:
    categorical_features = [
        column
        for column in customer_columns
        if column not in NUMERIC_FEATURES
    ]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", model)]
    )


def report(name, y_true, y_pred, y_proba) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    customer_columns = [
        column
        for column in data.columns
        if column not in {IDENTIFIER_COLUMN, TARGET_COLUMN}
    ]
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    X = data[customer_columns]
    y = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logistic_pipeline = make_pipeline(
        customer_columns,
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    random_forest_pipeline = make_pipeline(
        customer_columns,
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )

    logistic_pipeline.fit(X_train, y_train)
    random_forest_pipeline.fit(X_train, y_train)

    logistic_predictions = logistic_pipeline.predict(X_test)
    logistic_probabilities = logistic_pipeline.predict_proba(X_test)[:, 1]
    forest_predictions = random_forest_pipeline.predict(X_test)
    forest_probabilities = random_forest_pipeline.predict_proba(X_test)[:, 1]

    results = pd.DataFrame(
        [
            report(
                "Logistic Regression",
                y_test,
                logistic_predictions,
                logistic_probabilities,
            ),
            report(
                "Random Forest",
                y_test,
                forest_predictions,
                forest_probabilities,
            ),
        ]
    )

    print("=" * 60)
    print("MODEL COMPARISON (stratified 80/20 split)")
    print("=" * 60)
    print(results.to_string(index=False))
    print("\nDetailed classification report — Random Forest:")
    print(
        classification_report(
            y_test, forest_predictions, target_names=["Retained", "Churned"]
        )
    )

    preprocessing = random_forest_pipeline.named_steps["preprocessing"]
    feature_names = preprocessing.get_feature_names_out()
    importances = pd.Series(
        random_forest_pipeline.named_steps["classifier"].feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
    top_three = importances.head(3)

    print("=" * 60)
    print("TOP 3 PREDICTIVE FEATURES (Random Forest)")
    print("=" * 60)
    for feature, importance in top_three.items():
        print(f"{feature.removeprefix('numeric__').removeprefix('categorical__')}: {importance:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest_pipeline, MODEL_PATH, compress=3)
    print(f"\nSaved Random Forest pipeline -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
