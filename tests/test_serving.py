import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def client(mock_model):
    with patch("mlflow.sklearn.load_model", return_value=mock_model):
        from serving.app import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_payload():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 359.20,
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_expected_fields(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "threshold_used" in data


def test_predict_probability_range(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_above_threshold(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    data = response.json()
    # mock returns 0.7 probability, threshold is 0.5 → should predict True
    assert data["churn_prediction"] is True


def test_invalid_senior_citizen(client, sample_payload):
    sample_payload["SeniorCitizen"] = 5
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 422