import pytest
import pandas as pd
from sklearn.datasets import make_classification
from features.build_features import load_config
from training.train import load_model, evaluate


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def dummy_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        random_state=42,
        weights=[0.7, 0.3],
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


def test_load_model_returns_rf(config):
    model = load_model(config)
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


def test_load_model_params(config):
    model = load_model(config)
    assert model.n_estimators == config["model"]["params"]["n_estimators"]
    assert model.max_depth == config["model"]["params"]["max_depth"]


def test_evaluate_returns_all_metrics(config, dummy_data):
    X, y = dummy_data
    model = load_model(config)
    model.fit(X, y)
    metrics = evaluate(model, X, y, threshold=0.5)
    for key in ["roc_auc", "f1", "precision", "recall"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0