"""FastAPI service for customer churn predictions."""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "models" / "model_metadata.json"
logger = logging.getLogger(__name__)

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
    tenure: int = Field(ge=0, le=72, examples=[1])
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
    risk_level: Literal["low", "medium", "high"]
    if probability < cutoffs["low_upper"]:
        risk_level = "low"
    elif probability < cutoffs["medium_upper"]:
        risk_level = "medium"
    else:
        risk_level = "high"
    return {
        "churn_probability": probability,
        "risk_level": risk_level,
        "model_version": request.app.state.metadata["model_version"],
    }
