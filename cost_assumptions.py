"""Cost assumptions used to select customer outreach thresholds."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COST_CONFIG = PROJECT_DIR / "config" / "cost_assumptions.json"

# Illustrative assumptions, not measured campaign outcomes.
DEFAULT_OFFER_COST = 30.0
DEFAULT_CUSTOMER_VALUE = 500.0
DEFAULT_SAVE_RATE = 0.30


@dataclass(frozen=True)
class CostAssumptions:
    offer_cost: float
    customer_value: float
    save_rate: float

    @property
    def break_even_threshold(self) -> float:
        return self.offer_cost / (self.save_rate * self.customer_value)


def _validate(assumptions: CostAssumptions) -> None:
    values = {
        "offer_cost": assumptions.offer_cost,
        "customer_value": assumptions.customer_value,
        "save_rate": assumptions.save_rate,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Cost assumptions must be finite numbers.")
    if assumptions.offer_cost <= 0:
        raise ValueError("offer_cost must be greater than 0.")
    if assumptions.customer_value <= 0:
        raise ValueError("customer_value must be greater than 0.")
    if not 0 < assumptions.save_rate <= 1:
        raise ValueError("save_rate must be greater than 0 and at most 1.")
    if assumptions.offer_cost >= assumptions.save_rate * assumptions.customer_value:
        raise ValueError(
            "offer_cost must be less than save_rate * customer_value; "
            "otherwise no customer is worth contacting."
        )


def load_cost_assumptions(
    cost_config: Path = DEFAULT_COST_CONFIG,
    *,
    offer_cost: float | None = None,
    customer_value: float | None = None,
    save_rate: float | None = None,
) -> CostAssumptions:
    """Resolve defaults < JSON file < explicitly provided CLI values."""
    values = {
        "offer_cost": DEFAULT_OFFER_COST,
        "customer_value": DEFAULT_CUSTOMER_VALUE,
        "save_rate": DEFAULT_SAVE_RATE,
    }

    if cost_config.exists():
        if not cost_config.is_file():
            raise ValueError(f"Cost config is not a file: {cost_config}")
        try:
            file_values = json.loads(cost_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in cost config {cost_config}: {error}") from error
        if not isinstance(file_values, dict):
            raise ValueError("Cost config must contain a JSON object.")
        unknown = set(file_values) - set(values)
        if unknown:
            raise ValueError(
                "Unknown cost config keys: " + ", ".join(sorted(unknown))
            )
        values.update(file_values)

    overrides = {
        "offer_cost": offer_cost,
        "customer_value": customer_value,
        "save_rate": save_rate,
    }
    values.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    try:
        assumptions = CostAssumptions(
            offer_cost=float(values["offer_cost"]),
            customer_value=float(values["customer_value"]),
            save_rate=float(values["save_rate"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Cost assumptions must be numeric.") from error
    _validate(assumptions)
    return assumptions
