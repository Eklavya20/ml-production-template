import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

telco_schema = DataFrameSchema(
    {
        "tenure": Column(int, Check.greater_than_or_equal_to(0)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),
        "Contract": Column(str, Check.isin([
            "Month-to-month", "One year", "Two year"
        ])),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    coerce=True,
    strict=False  # allows extra columns
)