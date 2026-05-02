"""
degrade_model.py

Deliberately trains a bad model and registers it in MLflow.
Used to demonstrate ml-guardian's FAILED path — gates catch the regression
and block promotion to production.

Degradation strategy:
- Train on 10% of data (data starvation)
- n_estimators=1 (near-random forest)
"""

import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.build_features import load_config, prepare_data, build_preprocessor


def train_degraded(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="degraded_model_demo"):

        X, y, _ = prepare_data(config)

        # Degradation 1: use only 10% of training data
        X_small, _, y_small, _ = train_test_split(
            X, y, train_size=0.10, random_state=42, stratify=y
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_small, y_small,
            test_size=0.2,
            random_state=42,
            stratify=y_small,
        )

        preprocessor = build_preprocessor(config)

        # Degradation 2: near-useless model
        clf = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=42)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)

        # Log test data for ml-guardian
        X_test_full, _, y_test_full, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_full.to_csv("X_test.csv", index=False)
        y_test_full.to_csv("y_test.csv", index=False)
        mlflow.log_artifact("X_test.csv")
        mlflow.log_artifact("y_test.csv")
        os.remove("X_test.csv")
        os.remove("y_test.csv")

        # Log params so ml-guardian report shows what changed
        mlflow.log_params({
            "n_estimators": 1,
            "max_depth": 2,
            "training_data_fraction": 0.10,
            "degraded": True,
        })

        mlflow.set_tag("model_type", "degraded_demo")

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="telco_churn",
        )

        print("Degraded model registered. Run ml-guardian to see gates fail.")


if __name__ == "__main__":
    train_degraded()