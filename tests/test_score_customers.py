from pathlib import Path
import json

import pandas as pd

from score_customers import score_customers

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "scoring_customers.csv"


def test_score_customers_fixture(tmp_path, capsys) -> None:
    output_path = tmp_path / "ranked_customers.csv"
    result = score_customers(FIXTURE_PATH, output_path)

    written = pd.read_csv(output_path)
    assert list(written.columns) == [
        "customer_id",
        "churn_probability",
        "risk_level",
        "rank",
        "decile",
        "top_3_drivers",
    ]
    assert len(written) == 4
    assert written["rank"].tolist() == [1, 2, 3, 4]
    assert written["churn_probability"].is_monotonic_decreasing
    assert written["customer_id"].tolist() == result["customer_id"].tolist()
    drivers = json.loads(written.loc[0, "top_3_drivers"])
    assert len(drivers) == 3
    assert all(
        {"feature", "value", "reference_value", "probability_delta"}
        == set(driver)
        for driver in drivers
    )
    assert "Top-decile lift:" in capsys.readouterr().out
