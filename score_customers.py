"""Score and rank customers from a raw customer CSV file."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import METADATA_PATH, MODEL_PATH
from cost_assumptions import DEFAULT_COST_CONFIG, load_cost_assumptions
from model_utils import (
    counterfactual_driver_rows,
    oov_check,
    prepare_customer_features,
    risk_level_for,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--offer-cost", type=float)
    parser.add_argument("--customer-value", type=float)
    parser.add_argument("--save-rate", type=float)
    return parser.parse_args(argv)


def _target_values(data: pd.DataFrame) -> pd.Series | None:
    if "Churn" not in data:
        return None
    labels = data["Churn"]
    if labels.dtype == object or isinstance(labels.dtype, pd.StringDtype):
        labels = labels.map({"Yes": 1, "No": 0})
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Churn must contain only Yes/No or 0/1 values.")
    return labels.astype(int)


def _print_lift_and_gains(
    ranked: pd.DataFrame, target: pd.Series | None
) -> None:
    if target is None:
        print("Churn outcome column absent; top-decile lift and gains unavailable.")
        return

    ranked = ranked.copy()
    ranked["actual_churn"] = target.to_numpy()[ranked["_source_index"]]
    overall_rate = float(ranked["actual_churn"].mean())
    top_decile = ranked.loc[ranked["decile"] == 1, "actual_churn"]
    if overall_rate == 0:
        print("Top-decile lift unavailable because the input has no churn outcomes.")
    else:
        print(f"Top-decile lift: {top_decile.mean() / overall_rate:.3f}")

    gains_rows = []
    cumulative_customers = 0
    cumulative_churners = 0
    total_churners = int(ranked["actual_churn"].sum())
    for decile in range(1, 11):
        group = ranked.loc[ranked["decile"] == decile]
        cumulative_customers += len(group)
        cumulative_churners += int(group["actual_churn"].sum())
        gains_rows.append(
            {
                "decile": decile,
                "customers": len(group),
                "churners": int(group["actual_churn"].sum()),
                "cumulative_gain": (
                    cumulative_churners / total_churners
                    if total_churners
                    else None
                ),
                "cumulative_population": (
                    cumulative_customers / len(ranked) if len(ranked) else None
                ),
            }
        )
    print("Cumulative gains by ranked decile (decile 1 is highest risk):")
    print(pd.DataFrame(gains_rows).to_string(index=False))


def score_customers(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(input_path)
    if data.empty:
        raise ValueError("Input CSV contains no customer rows.")

    feature_list = metadata["feature_list"]
    missing_features = sorted(set(feature_list) - set(data.columns))
    if missing_features:
        raise ValueError(
            "Input CSV is missing model features: " + ", ".join(missing_features)
        )
    features = prepare_customer_features(data, feature_list)
    oov_counts = oov_check(features, model)
    print(f"OOV counts by categorical column: {oov_counts}")
    probabilities = model.predict_proba(features)[:, 1]
    top_drivers = counterfactual_driver_rows(
        model,
        features,
        probabilities,
        metadata["driver_reference_values"],
    )

    scored = pd.DataFrame(
        {
            "churn_probability": probabilities,
            "top_3_drivers": top_drivers,
            "risk_level": [
                risk_level_for(float(probability), metadata["risk_band_cutoffs"])
                for probability in probabilities
            ],
            "_source_index": np.arange(len(data)),
        }
    )
    scored = scored.sort_values(
        "churn_probability", ascending=False, kind="stable"
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["decile"] = ((scored["rank"] - 1) * 10 // len(scored)) + 1
    scored["decile"] = scored["decile"].clip(upper=10).astype(int)

    target = _target_values(data)
    _print_lift_and_gains(scored, target)

    customer_id_column = (
        "customer_id"
        if "customer_id" in data
        else "customerID"
        if "customerID" in data
        else None
    )
    output_columns = []
    if customer_id_column is not None:
        scored["customer_id"] = data[customer_id_column].to_numpy()[
            scored["_source_index"]
        ]
        output_columns.append("customer_id")
    output_columns.extend(
        [
            "churn_probability",
            "risk_level",
            "rank",
            "decile",
            "top_3_drivers",
        ]
    )
    result = scored[output_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved ranked customers -> {output_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
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
    score_customers(args.input, args.output)


if __name__ == "__main__":
    main()
