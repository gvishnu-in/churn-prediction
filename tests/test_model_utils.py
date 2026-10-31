import joblib
import numpy as np
import pandas as pd
import pytest

from app import CUSTOMER_EXAMPLE, MODEL_PATH
from model_utils import oov_check


def test_unseen_category_maps_to_dropped_baseline_and_is_counted() -> None:
    model = joblib.load(MODEL_PATH)
    baseline = pd.DataFrame([CUSTOMER_EXAMPLE])
    unseen = pd.DataFrame([{**CUSTOMER_EXAMPLE, "gender": "Unseen"}])

    counts = oov_check(unseen, model)
    assert counts["gender"] == 1
    assert all(count == 0 for column, count in counts.items() if column != "gender")

    fitted_pipeline = model.calibrated_classifiers_[0].estimator
    preprocessing = fitted_pipeline.named_steps["preprocessing"]
    with pytest.warns(UserWarning, match="unknown categories"):
        encoded = preprocessing.transform(
            pd.concat([baseline, unseen], ignore_index=True)
        )
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    assert np.array_equal(encoded[0], encoded[1])
