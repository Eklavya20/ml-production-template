import pandas as pd
import pytest
from src.features.schema import telco_schema
import pandera.pandas as pa

def test_valid_data_passes():
    df = pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [29.85],
        "TotalCharges": [359.20],
        "Contract": ["Month-to-month"],
        "Churn": ["No"]
    })
    telco_schema.validate(df)  # should not raise

def test_negative_tenure_fails():
    df = pd.DataFrame({
        "tenure": [-1],  # invalid
        "MonthlyCharges": [29.85],
        "TotalCharges": [359.20],
        "Contract": ["Month-to-month"],
        "Churn": ["No"]
    })
    with pytest.raises(pa.errors.SchemaErrors):
        telco_schema.validate(df, lazy=True)

def test_invalid_contract_type_fails():
    df = pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [29.85],
        "TotalCharges": [359.20],
        "Contract": ["Weekly"],  # not a valid contract
        "Churn": ["No"]
    })
    with pytest.raises(pa.errors.SchemaErrors):
        telco_schema.validate(df, lazy=True)