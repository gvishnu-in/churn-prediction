"""Shared helpers for model scoring and feature-vocabulary checks."""

from typing import Literal, Mapping


def risk_level_for(
    probability: float, cutoffs: Mapping[str, float]
) -> Literal["low", "medium", "high"]:
    if probability < cutoffs["low_upper"]:
        return "low"
    if probability < cutoffs["medium_upper"]:
        return "medium"
    return "high"
