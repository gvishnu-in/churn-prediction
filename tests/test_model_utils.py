import joblib
import numpy as np
import pandas as pd

from app import CUSTOMER_EXAMPLE, MODEL_PATH
from model_utils import oov_check, prepare_customer_features


def test_unseen_category_is_counted_and_does_not_alias_baseline() -> None:
    model = joblib.load(MODEL_PATH)
    baseline = pd.DataFrame([CUSTOMER_EXAMPLE])
    unseen = pd.DataFrame([{**CUSTOMER_EXAMPLE, "gender": "Unseen"}])

    counts = oov_check(unseen, model)
    assert counts["gender"] == 1
    assert all(count == 0 for column, count in counts.items() if column != "gender")

    fitted_pipeline = model.calibrated_classifiers_[0].estimator
    preprocessing = fitted_pipeline.named_steps["preprocessing"]
    encoded = preprocessing.transform(
        pd.concat([baseline, unseen], ignore_index=True)
    )
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    assert not np.array_equal(encoded[0], encoded[1])


def test_missing_category_is_counted_and_scored_by_saved_pipeline() -> None:
    model = joblib.load(MODEL_PATH)
    missing = pd.DataFrame([{**CUSTOMER_EXAMPLE, "gender": np.nan}])

    assert oov_check(missing, model)["gender"] == 1
    probability = float(model.predict_proba(missing)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_missing_numeric_value_is_imputed_by_saved_pipeline() -> None:
    model = joblib.load(MODEL_PATH)
    missing = pd.DataFrame([{**CUSTOMER_EXAMPLE, "TotalCharges": np.nan}])

    probability = float(model.predict_proba(missing)[0, 1])

    assert 0.0 <= probability <= 1.0


def test_cleaned_and_raw_features_normalize_to_the_same_contract() -> None:
    raw = pd.DataFrame([{**CUSTOMER_EXAMPLE, "SeniorCitizen": 0}])
    cleaned = raw.assign(SeniorCitizen="No", TotalCharges="85.7")

    raw_features = prepare_customer_features(raw)
    cleaned_features = prepare_customer_features(cleaned)

    pd.testing.assert_frame_equal(raw_features, cleaned_features)
