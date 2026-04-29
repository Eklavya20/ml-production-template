import logging
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, classification_report
)

from features.build_features import load_config, prepare_data, build_preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model(config: dict) -> RandomForestClassifier:
    params = config["model"]["params"]
    return RandomForestClassifier(**params)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }

    logger.info("\n" + classification_report(y_test, preds))
    return metrics


def train(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        # --- Data ---
        X, y, telco_prep = prepare_data(config)
        mlflow.set_tag("data_validation", "passed") 
        mlflow.log_param("input_rows", X.shape[0])  
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y,
        )

        # --- Pipeline ---
        preprocessor = build_preprocessor(config)
        clf = load_model(config)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # --- Cross-validation ---
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # --- Fit ---
        pipeline.fit(X_train, y_train)

        # --- Evaluate ---
        threshold = config["evaluation"]["threshold"]
        metrics = evaluate(pipeline, X_test, y_test, threshold)

        # ✅ --- Log test datasets as artifacts ---
        X_test.to_csv("X_test.csv", index=False)
        y_test.to_csv("y_test.csv", index=False)

        mlflow.log_artifact("X_test.csv")
        mlflow.log_artifact("y_test.csv")

        os.remove("X_test.csv")
        os.remove("y_test.csv")

        # --- Log to MLflow ---
        mlflow.log_params(config["model"]["params"])
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(config_path)

        # --- Register model ---
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="telco_churn",
        )

        logger.info(f"Run complete — ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        return pipeline, metrics


if __name__ == "__main__":
    train()