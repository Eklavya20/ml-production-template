import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_data_quality_metrics(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    metrics = {}

    # Missing values
    missing_rates = X.isnull().mean()
    total_missing = missing_rates.sum()
    metrics["dq_total_missing_rate"] = round(float(total_missing / len(X.columns)), 4)
    metrics["dq_cols_with_missing"] = int((missing_rates > 0).sum())

    # Class balance
    class_balance = y.value_counts(normalize=True)
    metrics["dq_class_balance_majority"] = round(float(class_balance.iloc[0]), 4)
    metrics["dq_class_balance_minority"] = round(float(class_balance.iloc[-1]), 4)

    # Numerical feature stats
    num_cols = config["features"]["numerical"]
    for col in num_cols:
        if col in X.columns:
            metrics[f"dq_mean_{col}"] = round(float(X[col].mean()), 4)
            metrics[f"dq_std_{col}"] = round(float(X[col].std()), 4)

    # Outliers via IQR
    outlier_counts = {}
    for col in num_cols:
        if col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum()
            outlier_counts[col] = int(outliers)

    metrics["dq_total_outliers"] = sum(outlier_counts.values())
    metrics["dq_row_count"] = len(X)

    logger.info(f"Data quality — missing rate: {metrics['dq_total_missing_rate']}, "
                f"outliers: {metrics['dq_total_outliers']}, "
                f"class balance: {metrics['dq_class_balance_minority']:.2%} minority")

    return metrics