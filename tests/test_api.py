import json

from fastapi.testclient import TestClient

from app import CUSTOMER_EXAMPLE, app


def test_health_and_prediction_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        docs = client.get("/docs")
        prediction = client.post("/predict", json=CUSTOMER_EXAMPLE)

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_version": app.state.metadata["model_version"],
    }
    assert docs.status_code == 200
    assert prediction.status_code == 200
    body = prediction.json()
    assert set(body) == {"churn_probability", "risk_level", "model_version"}
    assert body["model_version"] == app.state.metadata["model_version"]
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    cutoffs = app.state.metadata["risk_band_cutoffs"]
    expected_risk = (
        "low"
        if body["churn_probability"] < cutoffs["low_upper"]
        else "medium"
        if body["churn_probability"] < cutoffs["medium_upper"]
        else "high"
    )
    assert body["risk_level"] == expected_risk


def test_prediction_rejects_invalid_customer_fields() -> None:
    invalid_customer = {**CUSTOMER_EXAMPLE, "tenure": 121}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_prediction_rejects_category_outside_training_vocabulary() -> None:
    invalid_customer = {**CUSTOMER_EXAMPLE, "gender": "Other"}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_batch_prediction_happy_path() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json=[CUSTOMER_EXAMPLE, {**CUSTOMER_EXAMPLE, "tenure": 12}],
        )

    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) == 2
    assert all(
        prediction["model_version"] == app.state.metadata["model_version"]
        for prediction in predictions
    )
    assert all(
        0 <= prediction["churn_probability"] <= 1 for prediction in predictions
    )


def test_batch_prediction_enforces_maximum_size() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch", json=[CUSTOMER_EXAMPLE] * 1001
        )

    assert response.status_code == 422


def test_request_logs_are_structured_without_customer_fields(caplog) -> None:
    caplog.set_level("INFO", logger="app")
    with TestClient(app) as client:
        response = client.post("/predict", json=CUSTOMER_EXAMPLE)

    assert response.headers["X-Request-ID"]
    request_log = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "app" and record.getMessage().startswith("{")
    )
    assert {
        "request_id",
        "latency_ms",
        "model_version",
        "probability",
        "risk_level",
    }.issubset(request_log)
    assert "gender" not in request_log
    assert "customerID" not in request_log
