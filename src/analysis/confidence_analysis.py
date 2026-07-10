"""Confidence and per-class performance analysis for the MNIST CNN."""

import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

from src.analysis.error_analysis import generate_predictions, load_model_and_test_data
from src.utils.logging_config import setup_logger
from src.utils.paths import IMAGES_DIR, REPORTS_DIR


logger = setup_logger(__name__)

EXPLAINABILITY_REPORTS_DIR = REPORTS_DIR / "explainability"
EXPLAINABILITY_IMAGES_DIR = IMAGES_DIR / "explainability"
CONFIDENCE_ANALYSIS_PATH = EXPLAINABILITY_REPORTS_DIR / "confidence_analysis.csv"
CONFIDENCE_SUMMARY_PATH = EXPLAINABILITY_REPORTS_DIR / "confidence_summary.json"
PER_CLASS_PERFORMANCE_PATH = EXPLAINABILITY_REPORTS_DIR / "per_class_performance.csv"
CONFIDENCE_DISTRIBUTION_PATH = EXPLAINABILITY_IMAGES_DIR / "confidence_distribution.png"
CONFIDENCE_CORRECT_VS_INCORRECT_PATH = (
    EXPLAINABILITY_IMAGES_DIR / "confidence_correct_vs_incorrect.png"
)
PER_CLASS_ACCURACY_PATH = EXPLAINABILITY_IMAGES_DIR / "per_class_accuracy.png"


def ensure_output_dirs() -> None:
    """Create output directories for confidence-analysis artifacts."""
    EXPLAINABILITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _confidence_band(confidence: float) -> str:
    """Assign a confidence band using project thresholds."""
    if confidence < 0.70:
        return "low"
    if confidence < 0.90:
        return "medium"
    return "high"


def calculate_confidence_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    """Calculate summary confidence statistics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Full class probability matrix.

    Returns:
        Dictionary of confidence summary statistics.
    """
    confidence_scores = np.max(probabilities, axis=1)
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    incorrect_confidence = confidence_scores[incorrect_mask]
    return {
        "mean_confidence_overall": float(np.mean(confidence_scores)),
        "mean_confidence_correct": float(np.mean(confidence_scores[correct_mask])),
        "mean_confidence_incorrect": (
            float(np.mean(incorrect_confidence)) if incorrect_confidence.size else 0.0
        ),
        "median_confidence": float(np.median(confidence_scores)),
        "min_confidence": float(np.min(confidence_scores)),
        "max_confidence": float(np.max(confidence_scores)),
        "low_confidence_threshold": 0.70,
        "medium_confidence_threshold": 0.90,
        "number_of_low_confidence_predictions": int(np.sum(confidence_scores < 0.70)),
        "number_of_medium_confidence_predictions": int(
            np.sum((confidence_scores >= 0.70) & (confidence_scores < 0.90))
        ),
        "number_of_high_confidence_predictions": int(np.sum(confidence_scores >= 0.90)),
    }


def create_confidence_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create a per-sample confidence dataframe.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Full class probability matrix.

    Returns:
        Dataframe with confidence and correctness details.
    """
    confidence_scores = np.max(probabilities, axis=1)
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true), dtype=int),
            "true_label": y_true.astype(int),
            "predicted_label": y_pred.astype(int),
            "confidence": confidence_scores.astype(float),
            "correct": y_true == y_pred,
            "confidence_band": [_confidence_band(float(score)) for score in confidence_scores],
        }
    )


def compute_per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Compute per-class classification and confidence metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Full class probability matrix.

    Returns:
        Dataframe with support, accuracy, precision, recall, F1, and average
        confidence for each digit.
    """
    labels = np.arange(10)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows = []
    confidence_scores = np.max(probabilities, axis=1)
    for digit in labels:
        class_mask = y_true == digit
        rows.append(
            {
                "digit": int(digit),
                "support": int(support[digit]),
                "accuracy": float(np.mean(y_pred[class_mask] == y_true[class_mask])),
                "precision": float(precision[digit]),
                "recall": float(recall[digit]),
                "f1_score": float(f1[digit]),
                "average_confidence": float(np.mean(confidence_scores[class_mask])),
            }
        )
    return pd.DataFrame(rows)


def save_confidence_outputs(
    confidence_df: pd.DataFrame,
    confidence_summary: Dict[str, Any],
    per_class_df: pd.DataFrame,
) -> None:
    """Save confidence and per-class analysis tables.

    Args:
        confidence_df: Per-sample confidence dataframe.
        confidence_summary: Confidence statistics dictionary.
        per_class_df: Per-class performance dataframe.
    """
    ensure_output_dirs()
    confidence_df.to_csv(CONFIDENCE_ANALYSIS_PATH, index=False)
    CONFIDENCE_SUMMARY_PATH.write_text(
        json.dumps(confidence_summary, indent=2),
        encoding="utf-8",
    )
    per_class_df.to_csv(PER_CLASS_PERFORMANCE_PATH, index=False)
    logger.info("Saved confidence analysis outputs")


def save_confidence_plots(confidence_df: pd.DataFrame, per_class_df: pd.DataFrame) -> None:
    """Save confidence and per-class performance plots.

    Args:
        confidence_df: Per-sample confidence dataframe.
        per_class_df: Per-class performance dataframe.
    """
    ensure_output_dirs()

    plt.figure(figsize=(8, 5))
    sns.histplot(confidence_df["confidence"], bins=40, kde=True, color="#4C78A8")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CONFIDENCE_DISTRIBUTION_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=confidence_df,
        x="correct",
        y="confidence",
        hue="correct",
        palette={True: "#54A24B", False: "#E45756"},
        legend=False,
    )
    plt.title("Confidence: Correct vs Incorrect Predictions")
    plt.xlabel("Prediction Correct")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(CONFIDENCE_CORRECT_VS_INCORRECT_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=per_class_df, x="digit", y="accuracy", color="#72B7B2")
    plt.ylim(0.95, 1.0)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(PER_CLASS_ACCURACY_PATH, dpi=150)
    plt.close()
    logger.info("Saved confidence analysis plots")


def run_confidence_analysis() -> Dict[str, Any]:
    """Run confidence and per-class analysis.

    Returns:
        Confidence summary dictionary.
    """
    ensure_output_dirs()
    model, x_test, y_true, _ = load_model_and_test_data()
    y_pred, probabilities, _ = generate_predictions(model, x_test)
    confidence_summary = calculate_confidence_statistics(y_true, y_pred, probabilities)
    confidence_df = create_confidence_dataframe(y_true, y_pred, probabilities)
    per_class_df = compute_per_class_performance(y_true, y_pred, probabilities)
    save_confidence_outputs(confidence_df, confidence_summary, per_class_df)
    save_confidence_plots(confidence_df, per_class_df)
    return confidence_summary


def main() -> None:
    """Run the confidence-analysis workflow."""
    summary = run_confidence_analysis()
    logger.info(
        "Confidence analysis complete. Mean correct confidence: %.4f; "
        "mean incorrect confidence: %.4f",
        summary["mean_confidence_correct"],
        summary["mean_confidence_incorrect"],
    )


if __name__ == "__main__":
    main()
