"""
Compare churn classifiers, report feature importance, and persist the
Random Forest pipeline for scoring raw customer fields.

Run: python 03_classification_model.py
"""

from pathlib import Path
import argparse
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from cost_assumptions import DEFAULT_COST_CONFIG, CostAssumptions, load_cost_assumptions

RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_PRECISION = 0.60
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


def recall_at_precision(
    y_true: pd.Series, probabilities: np.ndarray, target_precision: float
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= target_precision)
    if len(eligible) == 0:
        return 0.0
    best = eligible[np.argmax(recall[:-1][eligible])]
    return float(recall[best])


def cross_validation_report(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float | str]:
    fold_results = []
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for train_indices, validation_indices in splitter.split(X, y):
        fold_model = deepcopy(pipeline)
        X_fold_train = X.iloc[train_indices]
        X_fold_validation = X.iloc[validation_indices]
        y_fold_train = y.iloc[train_indices]
        y_fold_validation = y.iloc[validation_indices]
        fold_model.fit(X_fold_train, y_fold_train)
        probabilities = fold_model.predict_proba(X_fold_validation)[:, 1]
        fold_results.append(
            {
                "roc_auc": roc_auc_score(y_fold_validation, probabilities),
                "pr_auc": average_precision_score(y_fold_validation, probabilities),
                "recall_at_precision": recall_at_precision(
                    y_fold_validation, probabilities, TARGET_PRECISION
                ),
            }
        )

    result: dict[str, float | str] = {"model": name}
    for metric in ("roc_auc", "pr_auc", "recall_at_precision"):
        values = np.array([fold[metric] for fold in fold_results])
        result[metric] = f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"
    return result


def select_cost_threshold(
    probabilities: np.ndarray, assumptions: CostAssumptions
) -> tuple[float, float]:
    thresholds = np.append(np.unique(probabilities), np.inf)
    expected_values = [
        probabilities[probabilities >= threshold].sum()
        * assumptions.save_rate
        * assumptions.customer_value
        - np.count_nonzero(probabilities >= threshold) * assumptions.offer_cost
        for threshold in thresholds
    ]
    best = int(np.argmax(expected_values))
    return float(thresholds[best]), float(expected_values[best])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--offer-cost", type=float)
    parser.add_argument("--customer-value", type=float)
    parser.add_argument("--save-rate", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assumptions = load_cost_assumptions(
        args.cost_config,
        offer_cost=args.offer_cost,
        customer_value=args.customer_value,
        save_rate=args.save_rate,
    )
    print(
        "COST ASSUMPTIONS (illustrative, not measured campaign results): "
        f"offer_cost=${assumptions.offer_cost:.2f}, "
        f"customer_value=${assumptions.customer_value:.2f}, "
        f"save_rate={assumptions.save_rate:.2f}, "
        f"break_even_threshold={assumptions.break_even_threshold:.3f}"
    )
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
    balanced_logistic_pipeline = make_pipeline(
        customer_columns,
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
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

    print("=" * 60)
    print(f"STRATIFIED {CV_FOLDS}-FOLD CV ON TRAINING SPLIT")
    print(
        "Mean ± sample standard deviation; recall is maximized at "
        f"precision >= {TARGET_PRECISION:.2f}."
    )
    cv_results = pd.DataFrame(
        [
            cross_validation_report("Logistic Regression", logistic_pipeline, X_train, y_train),
            cross_validation_report(
                "Logistic Regression (balanced)",
                balanced_logistic_pipeline,
                X_train,
                y_train,
            ),
            cross_validation_report(
                "Random Forest", random_forest_pipeline, X_train, y_train
            ),
        ]
    )
    print(cv_results.to_string(index=False))

    logistic_pipeline.fit(X_train, y_train)
    balanced_logistic_pipeline.fit(X_train, y_train)
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
                "Logistic Regression (balanced)",
                y_test,
                balanced_logistic_pipeline.predict(X_test),
                balanced_logistic_pipeline.predict_proba(X_test)[:, 1],
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

    X_fit, X_validation, y_fit, _ = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    threshold_model = make_pipeline(
        customer_columns,
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    )
    threshold_model.fit(X_fit, y_fit)
    validation_probabilities = threshold_model.predict_proba(X_validation)[:, 1]
    chosen_threshold, validation_net_value = select_cost_threshold(
        validation_probabilities, assumptions
    )
    holdout_flagged = forest_probabilities >= chosen_threshold
    print("=" * 60)
    print("COST-BASED THRESHOLD")
    print(f"Chosen validation threshold: {chosen_threshold:.6f}")
    print(f"Expected validation net value: ${validation_net_value:.2f}")
    print(f"Hold-out flagged customers: {int(holdout_flagged.sum())}")
    print(
        "Hold-out churners caught: "
        f"{int((holdout_flagged & (y_test.to_numpy() == 1)).sum())}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest_pipeline, MODEL_PATH, compress=3)
    print(f"\nSaved Random Forest pipeline -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
