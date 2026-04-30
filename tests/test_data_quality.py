import pandas as pd
from features.data_quality import compute_data_quality_metrics


def test_data_quality_metrics_keys():
    X = pd.DataFrame({
        "tenure": [12, 24, 0, 60],
        "MonthlyCharges": [29.85, 55.0, 20.0, 100.0],
        "TotalCharges": [359.0, 1320.0, 0.0, 6000.0],
    })
    y = pd.Series([0, 1, 0, 1])
    config = {"features": {"numerical": ["tenure", "MonthlyCharges", "TotalCharges"]}}

    metrics = compute_data_quality_metrics(X, y, config)

    assert "dq_total_missing_rate" in metrics
    assert "dq_class_balance_minority" in metrics
    assert "dq_total_outliers" in metrics
    assert "dq_row_count" in metrics
    assert metrics["dq_row_count"] == 4


def test_data_quality_detects_missing():
    X = pd.DataFrame({
        "tenure": [12, None, 0, 60],
        "MonthlyCharges": [29.85, 55.0, None, 100.0],
        "TotalCharges": [359.0, 1320.0, 0.0, 6000.0],
    })
    y = pd.Series([0, 1, 0, 1])
    config = {"features": {"numerical": ["tenure", "MonthlyCharges", "TotalCharges"]}}

    metrics = compute_data_quality_metrics(X, y, config)

    assert metrics["dq_cols_with_missing"] == 2
    assert metrics["dq_total_missing_rate"] > 0