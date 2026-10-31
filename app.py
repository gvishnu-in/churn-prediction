"""FastAPI service for customer churn predictions."""

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Annotated, Literal
import uuid

import joblib
import pandas as pd
import sklearn
from fastapi import Body, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from model_utils import oov_check, risk_level_for

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "models" / "model_metadata.json"
logger = logging.getLogger(__name__)


def _tenure_upper_bound() -> int:
    value = os.environ.get("TENURE_UPPER_BOUND", "120")
    try:
        upper_bound = int(value)
    except ValueError as error:
        raise RuntimeError("TENURE_UPPER_BOUND must be a non-negative integer.") from error
    if upper_bound < 0:
        raise RuntimeError("TENURE_UPPER_BOUND must be a non-negative integer.")
    return upper_bound


TENURE_UPPER_BOUND = _tenure_upper_bound()
# The training data tops out at 72 months; this bound admits longer legacy tenure
# while limiting extrapolation for values beyond the model's observed range.

CUSTOMER_EXAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.7,
    "TotalCharges": 85.7,
}


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [CUSTOMER_EXAMPLE]})

    gender: Literal["Female", "Male"] = Field(examples=["Female"])
    SeniorCitizen: Literal[0, 1] = Field(examples=[0])
    Partner: Literal["Yes", "No"] = Field(examples=["Yes"])
    Dependents: Literal["Yes", "No"] = Field(examples=["No"])
    tenure: int = Field(ge=0, le=TENURE_UPPER_BOUND, examples=[1])
    PhoneService: Literal["Yes", "No"] = Field(examples=["Yes"])
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        examples=["Fiber optic"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        examples=["Yes"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(examples=["Yes"])
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(ge=0, examples=[85.7])
    TotalCharges: float = Field(ge=0, examples=[85.7])


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = joblib.load(MODEL_PATH)
    application.state.metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    training_sklearn = application.state.metadata["versions"]["sklearn"]
    if sklearn.__version__ != training_sklearn:
        logger.warning(
            "MODEL SKLEARN VERSION MISMATCH: trained with %s, installed %s",
            training_sklearn,
            sklearn.__version__,
        )
    yield
    del application.state.model


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts churn probability from a customer's raw Telco fields.",
    version="1.0.0",
    lifespan=lifespan,
)


def _log_request(
    request: Request,
    *,
    request_id: str,
    started_at: float,
    status_code: int,
) -> None:
    metadata = getattr(request.app.state, "metadata", {})
    record = {
        "request_id": request_id,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "model_version": metadata.get("model_version"),
        "probability": getattr(request.state, "log_probability", None),
        "risk_level": getattr(request.state, "log_risk_level", None),
        "oov_counts": getattr(request.state, "oov_counts", {}),
        "status_code": status_code,
        "method": request.method,
        "path": request.url.path,
    }
    logger.info(json.dumps(record, separators=(",", ":")))


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _log_request(
            request,
            request_id=request_id,
            started_at=started_at,
            status_code=500,
        )
        raise
    _log_request(
        request,
        request_id=request_id,
        started_at=started_at,
        status_code=response.status_code,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        return {"status": "unavailable"}
    return {
        "status": "ok",
        "model_version": request.app.state.metadata["model_version"],
    }


@app.post("/predict")
def predict(
    customer: CustomerFeatures, request: Request
) -> dict[str, float | str]:
    features = pd.DataFrame([customer.model_dump()])
    probability = float(request.app.state.model.predict_proba(features)[0, 1])
    cutoffs = request.app.state.metadata["risk_band_cutoffs"]
    risk_level = risk_level_for(probability, cutoffs)
    request.state.log_probability = probability
    request.state.log_risk_level = risk_level
    request.state.oov_counts = oov_check(features, request.app.state.model)
    return {
        "churn_probability": probability,
        "risk_level": risk_level,
        "model_version": request.app.state.metadata["model_version"],
    }


@app.post("/predict/batch")
def predict_batch(
    customers: Annotated[
        list[CustomerFeatures], Body(min_length=1, max_length=1000)
    ],
    request: Request,
) -> list[dict[str, float | str]]:
    features = pd.DataFrame([customer.model_dump() for customer in customers])
    probabilities = request.app.state.model.predict_proba(features)[:, 1]
    metadata = request.app.state.metadata
    request.state.oov_counts = oov_check(features, request.app.state.model)
    predictions = []
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for probability in probabilities:
        value = float(probability)
        risk_level = risk_level_for(value, metadata["risk_band_cutoffs"])
        risk_counts[risk_level] += 1
        predictions.append(
            {
                "churn_probability": value,
                "risk_level": risk_level,
                "model_version": metadata["model_version"],
            }
        )
    request.state.log_probability = {
        "count": len(predictions),
        "mean": float(probabilities.mean()),
    }
    request.state.log_risk_level = risk_counts
    return predictions
