"""Compare a scoring CSV with the training data using population stability index."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
PSI_ALERT_THRESHOLD = 0.2
MISSING_BUCKET = ("__missing_value__",)


def _psi_from_counts(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    bin_count = len(reference_counts)
    reference_distribution = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * bin_count
    )
    current_distribution = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * bin_count
    )
    return float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(current_distribution / reference_distribution)
        )
    )


def numeric_psi(reference: pd.Series, current: pd.Series) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").to_numpy(
        dtype=float
    )
    current_values = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    reference_finite = np.isfinite(reference_values)
    current_finite = np.isfinite(current_values)
    if reference_finite.any():
        quantiles = np.quantile(
            reference_values[reference_finite], np.linspace(0, 1, 11)
        )
        boundaries = np.unique(quantiles)[1:-1]
    else:
        boundaries = np.array([], dtype=float)

    reference_bins = np.full(len(reference_values), -1, dtype=int)
    current_bins = np.full(len(current_values), -1, dtype=int)
    reference_bins[reference_finite] = np.digitize(
        reference_values[reference_finite], boundaries
    )
    current_bins[current_finite] = np.digitize(
        current_values[current_finite], boundaries
    )
    regular_bin_count = len(boundaries) + 1
    bins = [-1, *range(regular_bin_count)]
    return _psi_from_counts(
        np.array([(reference_bins == bucket).sum() for bucket in bins]),
        np.array([(current_bins == bucket).sum() for bucket in bins]),
    )


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    def normalized_values(values: pd.Series) -> list[object]:
        return [
            MISSING_BUCKET if pd.isna(value) else str(value)
            for value in values.tolist()
        ]

    reference_values = normalized_values(reference)
    current_values = normalized_values(current)
    categories = sorted(
        set(reference_values) | set(current_values),
        key=lambda value: repr(value),
    )
    return _psi_from_counts(
        np.array([reference_values.count(category) for category in categories]),
        np.array([current_values.count(category) for category in categories]),
    )


def compute_psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = PSI_ALERT_THRESHOLD,
) -> pd.DataFrame:
    if reference.empty or current.empty:
        raise ValueError("Reference and scoring data must both contain rows.")
    feature_columns = [
        column
        for column in reference.columns
        if column not in {"customerID", "customer_id", "Churn"}
    ]
    missing_columns = sorted(set(feature_columns) - set(current.columns))
    if missing_columns:
        raise ValueError(
            "Scoring data is missing reference features: "
            + ", ".join(missing_columns)
        )

    rows = []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            feature_type = "numeric"
            psi = numeric_psi(reference[column], current[column])
        else:
            feature_type = "categorical"
            psi = categorical_psi(reference[column], current[column])
        rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "psi": psi,
                "drift_flag": psi > threshold,
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=PSI_ALERT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.input)
    table = compute_psi_table(reference, current, args.threshold)
    print(table.to_string(index=False))
    print(f"Drift is flagged when PSI > {args.threshold:.3f}.")


if __name__ == "__main__":
    main()
