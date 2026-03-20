import logging
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- Schemas ---

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    @field_validator("SeniorCitizen")
    @classmethod
    def senior_citizen_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")
        return v


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- App state ---

app_state = {}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    try:
        model_uri = config["serving"]["model_uri"]
        app_state["model"] = mlflow.sklearn.load_model(model_uri)
        app_state["threshold"] = config["evaluation"]["threshold"]
        logger.info(f"Model loaded from {model_uri}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app_state["model"] = None

    yield

    # Shutdown
    app_state.clear()


# --- App ---

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production ML serving layer for the telco churn model.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Routes ---

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=app_state.get("model") is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([customer.model_dump()])
    threshold = app_state["threshold"]

    try:
        proba = model.predict_proba(input_df)[:, 1][0]
        prediction = bool(proba >= threshold)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=prediction,
        threshold_used=threshold,
    )


@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API — visit /docs for the Swagger UI"}