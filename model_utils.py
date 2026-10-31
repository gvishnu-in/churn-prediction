"""Shared helpers for model scoring and feature-vocabulary checks."""

from typing import Literal, Mapping

import pandas as pd


def risk_level_for(
    probability: float, cutoffs: Mapping[str, float]
) -> Literal["low", "medium", "high"]:
    if probability < cutoffs["low_upper"]:
        return "low"
    if probability < cutoffs["medium_upper"]:
        return "medium"
    return "high"


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
