import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["raw_path"])
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


class TelcoPreprocessor(BaseEstimator, TransformerMixin):
    """Handles Telco-specific quirks before the sklearn pipeline."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # TotalCharges is string with whitespace for new customers
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"] = X["TotalCharges"].fillna(0.0)

        # Binary yes/no columns → int
        binary_cols = [
            col for col in X.columns
            if X[col].dropna().isin(["Yes", "No"]).all()
        ]
        for col in binary_cols:
            X[col] = (X[col] == "Yes").astype(int)

        return X


def encode_target(series: pd.Series) -> pd.Series:
    return (series == "Yes").astype(int)


def build_preprocessor(config: dict) -> ColumnTransformer:
    cat_cols = [
        col for col in config["features"]["categorical"]
        if col not in _binary_sentinel()
    ]
    num_cols = config["features"]["numerical"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
    )


def _binary_sentinel() -> list:
    """Columns already binarised by TelcoPreprocessor — skip re-encoding."""
    return [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    ]


def prepare_data(config: dict):
    df = load_data(config)

    drop_cols = config["features"]["drop"]
    target_col = config["data"]["target_column"]

    y = encode_target(df[target_col])
    X = df.drop(columns=drop_cols + [target_col])

    telco_prep = TelcoPreprocessor()
    X = telco_prep.fit_transform(X)

    logger.info(f"Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    return X, y, telco_prep