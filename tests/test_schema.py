import pandas as pd
import pytest
from src.features.schema import telco_schema
import pandera.pandas as pa


def test_valid_data_passes():
    df = pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [29.85],
        "TotalCharges": [359.20],
    })
    telco_schema.validate(df)


def test_negative_tenure_fails():
    df = pd.DataFrame({
        "tenure": [-1],
        "MonthlyCharges": [29.85],
        "TotalCharges": [359.20],
    })
    with pytest.raises(pa.errors.SchemaErrors):
        telco_schema.validate(df, lazy=True)


def test_negative_monthly_charges_fails():
    df = pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [-10.0],
        "TotalCharges": [359.20],
    })
    with pytest.raises(pa.errors.SchemaErrors):
        telco_schema.validate(df, lazy=True)