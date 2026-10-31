"""
Compare churn classifiers, report feature importance, and persist the
Random Forest pipeline for scoring raw customer fields.

Run: python 03_classification_model.py
"""

from pathlib import Path
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
MODEL_VERSION = "1.1.0"
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
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1))
    return result


def calibrated_oof_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
) -> tuple[np.ndarray, list[float]]:
    probabilities = np.empty(len(y), dtype=float)
    fold_briers = []
    outer_cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    for train_indices, validation_indices in outer_cv.split(X, y):
        calibrated = CalibratedClassifierCV(
            estimator=deepcopy(pipeline),
            method=method,
            cv=inner_cv,
        )
        calibrated.fit(X.iloc[train_indices], y.iloc[train_indices])
        validation_probabilities = calibrated.predict_proba(
            X.iloc[validation_indices]
        )[:, 1]
        probabilities[validation_indices] = validation_probabilities
        fold_briers.append(
            float(
                brier_score_loss(
                    y.iloc[validation_indices], validation_probabilities
                )
            )
        )
    return probabilities, fold_briers


def reliability_summary(
    y_true: pd.Series, probabilities: np.ndarray
) -> list[dict[str, float | int | None]]:
    bins = np.minimum((probabilities * 10).astype(int), 9)
    summary = []
    for bin_number in range(10):
        selected = bins == bin_number
        count = int(selected.sum())
        summary.append(
            {
                "bin": f"{bin_number / 10:.1f}-{(bin_number + 1) / 10:.1f}",
                "count": count,
                "mean_predicted_probability": (
                    float(probabilities[selected].mean()) if count else None
                ),
                "observed_churn_rate": (
                    float(y_true.to_numpy()[selected].mean()) if count else None
                ),
            }
        )
    return summary


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
        "Mean +/- sample standard deviation; recall is maximized at "
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
    cv_display = pd.DataFrame(
        {
            "model": cv_results["model"],
            "roc_auc": cv_results.apply(
                lambda row: f"{row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}",
                axis=1,
            ),
            "pr_auc": cv_results.apply(
                lambda row: f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f}",
                axis=1,
            ),
            "recall_at_precision": cv_results.apply(
                lambda row: (
                    f"{row['recall_at_precision_mean']:.3f} +/- "
                    f"{row['recall_at_precision_std']:.3f}"
                ),
                axis=1,
            ),
        }
    )
    print(cv_display.to_string(index=False))

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

    print("=" * 60)
    print("CALIBRATION SELECTION (stratified validation folds)")
    calibration_results = {}
    calibration_probabilities = {}
    for method in ("isotonic", "sigmoid"):
        oof_probabilities, fold_briers = calibrated_oof_probabilities(
            random_forest_pipeline, X_train, y_train, method
        )
        calibration_results[method] = {
            "fold_brier_scores": fold_briers,
            "mean_brier_score": float(np.mean(fold_briers)),
            "std_brier_score": float(np.std(fold_briers, ddof=1)),
        }
        calibration_probabilities[method] = oof_probabilities
        print(
            f"{method}: Brier score {np.mean(fold_briers):.4f} "
            f"+/- {np.std(fold_briers, ddof=1):.4f}"
        )
    selected_calibration = min(
        calibration_results, key=lambda method: calibration_results[method]["mean_brier_score"]
    )
    validation_probabilities = calibration_probabilities[selected_calibration]
    calibrated_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            customer_columns,
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        method=selected_calibration,
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    before_brier = float(brier_score_loss(y_test, forest_probabilities))
    after_brier = float(brier_score_loss(y_test, calibrated_probabilities))
    print(f"Selected calibration: {selected_calibration}")
    print(f"Hold-out Brier before calibration: {before_brier:.4f}")
    print(f"Hold-out Brier after calibration:  {after_brier:.4f}")
    print("Reliability summary before calibration (10 equal-width probability bins):")
    reliability_before = reliability_summary(y_test, forest_probabilities)
    print(pd.DataFrame(reliability_before).to_string(index=False))
    print("Reliability summary after calibration (10 equal-width probability bins):")
    reliability_after = reliability_summary(y_test, calibrated_probabilities)
    print(pd.DataFrame(reliability_after).to_string(index=False))

    chosen_threshold, validation_net_value = select_cost_threshold(
        validation_probabilities, assumptions
    )
    risk_low_upper, risk_medium_upper = np.quantile(
        validation_probabilities, [1 / 3, 2 / 3]
    )
    holdout_flagged = calibrated_probabilities >= chosen_threshold
    print("=" * 60)
    print("COST-BASED THRESHOLD")
    print(f"Chosen validation threshold: {chosen_threshold:.6f}")
    print(f"Expected validation net value: ${validation_net_value:.2f}")
    print(f"Hold-out flagged customers: {int(holdout_flagged.sum())}")
    print(
        "Hold-out churners caught: "
        f"{int((holdout_flagged & (y_test.to_numpy() == 1)).sum())}"
    )
    print(
        "Risk-band cutoffs (training out-of-fold calibrated probability tertiles): "
        f"low < {risk_low_upper:.6f}, medium < {risk_medium_upper:.6f}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH, compress=3)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "training_row_count": int(len(X_train)),
        "feature_list": customer_columns,
        "chosen_threshold": chosen_threshold,
        "risk_band_cutoffs": {
            "low_upper": float(risk_low_upper),
            "medium_upper": float(risk_medium_upper),
        },
        "cost_assumptions": {
            "offer_cost": assumptions.offer_cost,
            "customer_value": assumptions.customer_value,
            "save_rate": assumptions.save_rate,
            "break_even_threshold": assumptions.break_even_threshold,
        },
        "evaluation_metrics": {
            "cross_validation": cv_results.to_dict(orient="records"),
            "holdout": results.to_dict(orient="records"),
            "calibration": {
                "selected_method": selected_calibration,
                "validation_brier_scores": calibration_results,
                "holdout_brier_before": before_brier,
                "holdout_brier_after": after_brier,
                "reliability_before": reliability_before,
                "reliability_after": reliability_after,
            },
            "threshold": {
                "expected_validation_net_value": validation_net_value,
                "holdout_flagged_customers": int(holdout_flagged.sum()),
                "holdout_churners_caught": int(
                    (holdout_flagged & (y_test.to_numpy() == 1)).sum()
                ),
            },
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nSaved calibrated Random Forest pipeline -> {MODEL_PATH}")
    print(f"Saved model metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
