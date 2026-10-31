import json

import joblib
import pandas as pd
import pytest

from app import CUSTOMER_EXAMPLE, MODEL_PATH, METADATA_PATH

GOLDEN_PROBABILITY = 0.757015577196797


def test_golden_customer_probability_and_metadata() -> None:
    model = joblib.load(MODEL_PATH)
    probability = float(
        model.predict_proba(pd.DataFrame([CUSTOMER_EXAMPLE]))[0, 1]
    )

    assert probability == pytest.approx(GOLDEN_PROBABILITY, abs=1e-8)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["model_version"]
    assert metadata["cost_assumptions"] == {
        "offer_cost": 30.0,
        "customer_value": 500.0,
        "save_rate": 0.3,
        "break_even_threshold": 0.2,
    }
