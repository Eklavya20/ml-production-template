import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prefect import flow, task, get_run_logger
from sklearn.model_selection import train_test_split

from features.build_features import load_config, prepare_data, build_preprocessor
from training.train import load_model, evaluate
from evaluation.evaluate import run_full_evaluation

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


@task(name="load-and-prepare-data", retries=2, retry_delay_seconds=5)
def task_prepare_data(config: dict):
    logger = get_run_logger()
    X, y, telco_prep = prepare_data(config)
    logger.info(f"Data prepared — {X.shape[0]} rows, {X.shape[1]} cols")
    return X, y, telco_prep


@task(name="split-data")
def task_split_data(X, y, config: dict):
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@task(name="build-pipeline")
def task_build_pipeline(config: dict):
    logger = get_run_logger()
    preprocessor = build_preprocessor(config)
    clf = load_model(config)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    logger.info("Pipeline built successfully")
    return pipeline


@task(name="train-model")
def task_train(pipeline, X_train, y_train):
    logger = get_run_logger()
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete")
    return pipeline


@task(name="evaluate-model")
def task_evaluate(pipeline, X_test, y_test, config: dict):
    logger = get_run_logger()
    threshold = config["evaluation"]["threshold"]
    metrics = evaluate(pipeline, X_test, y_test, threshold)
    logger.info(f"Evaluation complete — ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics


@task(name="run-full-evaluation")
def task_full_evaluation(pipeline, X_test, y_test, config: dict):
    logger = get_run_logger()
    optimal_threshold = run_full_evaluation(pipeline, X_test, y_test, config)
    logger.info(f"Full evaluation done — optimal threshold: {optimal_threshold:.2f}")
    return optimal_threshold


@task(name="log-and-register-model")
def task_register_model(pipeline, metrics: dict, config: dict, X_test, y_test):
    logger = get_run_logger()
    mlflow.log_params(config["model"]["params"])
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("configs/config.yaml")

    # Log test data for ml-guardian
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    mlflow.log_artifact("X_test.csv")
    mlflow.log_artifact("y_test.csv")
    os.remove("X_test.csv")
    os.remove("y_test.csv")

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="telco_churn",
    )
    logger.info("Model logged and registered in MLflow")


@flow(name="telco-churn-training-pipeline", log_prints=True)
def training_pipeline(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        # Data
        X, y, _ = task_prepare_data(config)
        X_train, X_test, y_train, y_test = task_split_data(X, y, config)

        # Train
        pipeline = task_build_pipeline(config)
        trained_pipeline = task_train(pipeline, X_train, y_train)

        # Evaluate
        metrics = task_evaluate(trained_pipeline, X_test, y_test, config)
        task_full_evaluation(trained_pipeline, X_test, y_test, config)

        # Register
        task_register_model(trained_pipeline, metrics, config, X_test, y_test)


if __name__ == "__main__":
    training_pipeline()