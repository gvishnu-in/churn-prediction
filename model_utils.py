"""Shared helpers for model scoring and feature-vocabulary checks."""

import json
from typing import Literal, Mapping

import numpy as np
import pandas as pd

CUSTOMER_FEATURES = (
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
)
NUMERIC_FEATURES = (
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)


def risk_level_for(
    probability: float, cutoffs: Mapping[str, float]
) -> Literal["low", "medium", "high"]:
    if probability < cutoffs["low_upper"]:
        return "low"
    if probability < cutoffs["medium_upper"]:
        return "medium"
    return "high"


def prepare_customer_features(
    data: pd.DataFrame, feature_list: tuple[str, ...] | list[str] = CUSTOMER_FEATURES
) -> pd.DataFrame:
    """Normalize cleaned or raw CSV fields into the API/model input contract."""
    missing_features = sorted(set(feature_list) - set(data.columns))
    if missing_features:
        raise ValueError(
            "Input data is missing model features: " + ", ".join(missing_features)
        )

    features = data.loc[:, list(feature_list)].copy()
    if "SeniorCitizen" in features:
        features["SeniorCitizen"] = features["SeniorCitizen"].replace(
            {"No": 0, "Yes": 1}
        )
    for column in feature_list:
        if column in NUMERIC_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
        else:
            features[column] = features[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return features


def oov_check(data: pd.DataFrame, model) -> dict[str, int]:
    """Count categorical values not present in the model's fitted vocabulary."""
    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        fitted_pipeline = calibrated_classifiers[0].estimator
    else:
        fitted_pipeline = model

    try:
        preprocessing = fitted_pipeline.named_steps["preprocessing"]
        categorical_transformer = next(
            transformer
            for name, transformer, _ in preprocessing.transformers_
            if name == "categorical"
        )
        categorical_columns = next(
            columns
            for name, _, columns in preprocessing.transformers_
            if name == "categorical"
        )
        encoder = (
            categorical_transformer.named_steps["encoder"]
            if hasattr(categorical_transformer, "named_steps")
            else categorical_transformer
        )
        vocabularies = encoder.categories_
    except (AttributeError, KeyError, StopIteration, IndexError) as error:
        raise ValueError(
            "The supplied model does not expose a fitted categorical vocabulary."
        ) from error

    return {
        column: int((~data[column].isin(categories)).sum())
        for column, categories in zip(
            categorical_columns, vocabularies, strict=True
        )
    }


def counterfactual_driver_rows(
    model,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    reference_values: Mapping[str, object],
    *,
    top_n: int = 3,
) -> list[str]:
    """Return isolated feature-replacement probability deltas as JSON cells."""
    missing_references = sorted(set(features.columns) - set(reference_values))
    if missing_references:
        raise ValueError(
            "Model metadata is missing driver reference values for: "
            + ", ".join(missing_references)
        )

    columns = list(features.columns)
    feature_deltas = np.empty((len(columns), len(features)), dtype=float)
    for batch_start in range(0, len(features), 512):
        batch_end = min(batch_start + 512, len(features))
        batch = features.iloc[batch_start:batch_end]
        counterfactuals = []
        for column in columns:
            counterfactual = batch.copy()
            counterfactual[column] = reference_values[column]
            counterfactuals.append(counterfactual)
        counterfactual_probabilities = model.predict_proba(
            pd.concat(counterfactuals, ignore_index=True)
        )[:, 1].reshape(len(columns), len(batch))
        feature_deltas[:, batch_start:batch_end] = (
            probabilities[batch_start:batch_end] - counterfactual_probabilities
        )

    output = []
    for row_index in range(len(features)):
        drivers = []
        for column_index, column in enumerate(columns):
            value = features.iloc[row_index][column]
            drivers.append(
                {
                    "feature": column,
                    "value": (
                        None
                        if pd.isna(value)
                        else value.item()
                        if isinstance(value, np.generic)
                        else value
                    ),
                    "reference_value": reference_values[column],
                    "probability_delta": round(
                        float(feature_deltas[column_index, row_index]), 6
                    ),
                }
            )
        drivers.sort(key=lambda driver: abs(driver["probability_delta"]), reverse=True)
        output.append(json.dumps(drivers[:top_n], allow_nan=False))
    return output
