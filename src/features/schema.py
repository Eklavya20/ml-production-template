import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

telco_schema = DataFrameSchema(
    {
        "tenure": Column(int, Check.greater_than_or_equal_to(0)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),
    },
    coerce=True,
    strict=False  # allows extra columns
)
def validate_input(df):
    try:
        telco_schema.validate(df, lazy=True)
        print("✅ Schema validation passed")
        return True
    except pa.errors.SchemaErrors as e:
        print("❌ Schema validation failed:")
        print(e.failure_cases)
        raise