import json
from pathlib import Path

import pytest

from cost_assumptions import (
    DEFAULT_CUSTOMER_VALUE,
    DEFAULT_OFFER_COST,
    DEFAULT_SAVE_RATE,
    load_cost_assumptions,
)


def test_cost_assumption_precedence(tmp_path) -> None:
    config = tmp_path / "cost_assumptions.json"
    config.write_text(
        json.dumps(
            {"offer_cost": 40, "customer_value": 600, "save_rate": 0.25}
        ),
        encoding="utf-8",
    )

    assumptions = load_cost_assumptions(
        config, offer_cost=50, customer_value=None, save_rate=0.4
    )

    assert assumptions.offer_cost == 50
    assert assumptions.customer_value == 600
    assert assumptions.save_rate == 0.4
    defaults = load_cost_assumptions(tmp_path / "absent.json")
    assert defaults.offer_cost == DEFAULT_OFFER_COST
    assert defaults.customer_value == DEFAULT_CUSTOMER_VALUE
    assert defaults.save_rate == DEFAULT_SAVE_RATE


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"offer_cost": 0}, "offer_cost"),
        ({"customer_value": -1}, "customer_value"),
        ({"save_rate": 0}, "save_rate"),
        ({"save_rate": 1.1}, "save_rate"),
        ({"offer_cost": 150, "customer_value": 500, "save_rate": 0.3}, "no customer"),
    ],
)
def test_invalid_cost_assumptions_are_rejected(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_cost_assumptions(
            cost_config=Path("missing-cost-config.json"), **overrides
        )


def test_boolean_and_string_cost_values_are_rejected(tmp_path) -> None:
    config = tmp_path / "cost_assumptions.json"
    config.write_text(
        json.dumps(
            {"offer_cost": True, "customer_value": 500, "save_rate": 0.3}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="offer_cost must be a number"):
        load_cost_assumptions(config)

    with pytest.raises(ValueError, match="offer_cost must be a number"):
        load_cost_assumptions(
            tmp_path / "absent.json", offer_cost="30"
        )
