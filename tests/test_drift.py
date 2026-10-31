import pandas as pd

from monitoring.drift import compute_psi_table


def test_psi_flags_synthetic_numeric_and_categorical_shift() -> None:
    reference = pd.DataFrame(
        {
            "tenure": list(range(100)),
            "Contract": ["Month-to-month"] * 90 + ["One year"] * 10,
        }
    )
    current = pd.DataFrame(
        {
            "tenure": list(range(200, 300)),
            "Contract": ["Month-to-month"] * 10 + ["One year"] * 90,
        }
    )

    result = compute_psi_table(reference, current)

    assert set(result["feature_type"]) == {"numeric", "categorical"}
    assert (result["psi"] > 0.2).all()
    assert result["drift_flag"].all()
