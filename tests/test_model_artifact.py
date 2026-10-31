import json

import joblib
import pandas as pd
import pytest

from app import CUSTOMER_EXAMPLE, MODEL_PATH, METADATA_PATH

GOLDEN_PROBABILITY = 0.8310408295004077


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
    assert set(metadata["driver_reference_values"]) == set(metadata["feature_list"])
    band_rates = metadata["risk_band_validation_summary"]
    assert band_rates["low"]["observed_churn_rate"] < band_rates["medium"]["observed_churn_rate"]
    assert band_rates["medium"]["observed_churn_rate"] < band_rates["high"]["observed_churn_rate"]
