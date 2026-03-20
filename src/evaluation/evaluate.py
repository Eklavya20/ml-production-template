import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay, classification_report
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_optimal_threshold(y_true, proba, metric: str = "f1") -> float:
    """Sweep thresholds and return the one maximising the chosen metric."""
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []

    for t in thresholds:
        preds = (proba >= t).astype(int)
        if metric == "f1":
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_true, preds, zero_division=0))
        elif metric == "precision":
            from sklearn.metrics import precision_score
            scores.append(precision_score(y_true, preds, zero_division=0))
        elif metric == "recall":
            from sklearn.metrics import recall_score
            scores.append(recall_score(y_true, preds, zero_division=0))

    best_threshold = thresholds[np.argmax(scores)]
    logger.info(f"Optimal threshold ({metric}): {best_threshold:.2f} → score: {max(scores):.4f}")
    return float(best_threshold)


def plot_roc_curve(y_true, proba, save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")
    return fig


def plot_precision_recall_curve(y_true, proba, save_path: str = None):
    precision, recall, _ = precision_recall_curve(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"PR curve saved to {save_path}")
    return fig


def plot_calibration_curve(y_true, proba, save_path: str = None):
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, proba, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Calibration curve saved to {save_path}")
    return fig


def plot_confusion_matrix(y_true, preds, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    return fig


def run_full_evaluation(pipeline, X_test, y_test, config: dict):
    """Run all evaluation plots and log them to the active MLflow run."""
    import os
    os.makedirs("reports", exist_ok=True)

    proba = pipeline.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, proba, metric="f1")
    preds = (proba >= optimal_threshold).astype(int)

    logger.info("\n" + classification_report(y_test, preds))

    plots = {
        "reports/roc_curve.png": plot_roc_curve(y_test, proba, "reports/roc_curve.png"),
        "reports/pr_curve.png": plot_precision_recall_curve(y_test, proba, "reports/pr_curve.png"),
        "reports/calibration_curve.png": plot_calibration_curve(y_test, proba, "reports/calibration_curve.png"),
        "reports/confusion_matrix.png": plot_confusion_matrix(y_test, preds, "reports/confusion_matrix.png"),
    }

    for path in plots:
        mlflow.log_artifact(path)
        logger.info(f"Logged artifact: {path}")

    mlflow.log_metric("optimal_threshold", optimal_threshold)
    plt.close("all")

    return optimal_threshold