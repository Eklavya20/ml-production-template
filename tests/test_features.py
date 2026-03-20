import pytest
import pandas as pd
from features.build_features import TelcoPreprocessor, encode_target, load_config


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["1234-ABCD", "5678-EFGH"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["359.20", " "],  # whitespace mimics raw data
        "Churn": ["No", "Yes"],
    })


def test_telco_preprocessor_total_charges(sample_df):
    prep = TelcoPreprocessor()
    result = prep.fit_transform(sample_df)
    assert result["TotalCharges"].dtype == float
    assert result["TotalCharges"].isna().sum() == 0


def test_telco_preprocessor_binary_columns(sample_df):
    prep = TelcoPreprocessor()
    result = prep.fit_transform(sample_df)
    assert set(result["Partner"].unique()).issubset({0, 1})
    assert set(result["Dependents"].unique()).issubset({0, 1})


def test_encode_target():
    series = pd.Series(["Yes", "No", "Yes", "No"])
    result = encode_target(series)
    assert list(result) == [1, 0, 1, 0]


def test_load_config():
    config = load_config("configs/config.yaml")
    assert "data" in config
    assert "model" in config
    assert "features" in config
    assert "mlflow" in config