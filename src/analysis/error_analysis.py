"""Error analysis utilities for the trained MNIST CNN model."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

from src.data.data_loader import load_mnist_data
from src.inference.predict import load_trained_model
from src.preprocessing.image_preprocessor import preprocess_mnist_images
from src.utils.logging_config import setup_logger
from src.utils.paths import IMAGES_DIR, REPORTS_DIR


logger = setup_logger(__name__)

ERROR_REPORTS_DIR = REPORTS_DIR / "error_analysis"
ERROR_IMAGES_DIR = IMAGES_DIR / "error_analysis"
MISCLASSIFIED_REPORT_PATH = ERROR_REPORTS_DIR / "misclassified_samples.csv"
CONFUSION_PAIRS_PATH = ERROR_REPORTS_DIR / "top_confusion_pairs.csv"
MISCLASSIFIED_GRID_PATH = ERROR_IMAGES_DIR / "misclassified_examples.png"
CORRECT_GRID_PATH = ERROR_IMAGES_DIR / "high_confidence_correct_predictions.png"


def ensure_output_dirs() -> None:
    """Create output directories for error-analysis artifacts."""
    ERROR_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_test_data() -> Tuple[Model, np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved model and preprocessed MNIST test data.

    Returns:
        Tuple containing the model, processed test images, raw test labels, and
        raw test images.
    """
    logger.info("Loading saved model and MNIST test data")
    model = load_trained_model()
    (_, _), (x_test_raw, y_test_raw) = load_mnist_data()
    x_test_processed = preprocess_mnist_images(x_test_raw)
    return model, x_test_processed, y_test_raw, x_test_raw


def generate_predictions(model: Model, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate class predictions and confidence scores.

    Args:
        model: Trained Keras classification model.
        x_test: Processed test images.

    Returns:
        Tuple of predicted labels, class probabilities, and confidence scores.
    """
    logger.info("Generating predictions for %s test samples", x_test.shape[0])
    class_probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(class_probabilities, axis=1)
    confidence_scores = np.max(class_probabilities, axis=1)
    return predicted_labels, class_probabilities, confidence_scores


def get_misclassified_indices(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return indices where predicted labels differ from true labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Array of misclassified sample indices.
    """
    return np.where(y_true != y_pred)[0]


def build_misclassification_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_scores: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build a dataframe describing every misclassified sample.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidence_scores: Maximum predicted probability per sample.
        probabilities: Full class probability matrix.

    Returns:
        Dataframe with one row per misclassified sample.
    """
    misclassified_indices = get_misclassified_indices(y_true, y_pred)
    rows = []
    for sample_index in misclassified_indices:
        true_label = int(y_true[sample_index])
        predicted_label = int(y_pred[sample_index])
        rows.append(
            {
                "sample_index": int(sample_index),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(confidence_scores[sample_index]),
                "true_class_probability": float(probabilities[sample_index, true_label]),
                "predicted_class_probability": float(probabilities[sample_index, predicted_label]),
            }
        )
    return pd.DataFrame(rows)


def save_misclassification_report(df: pd.DataFrame) -> Path:
    """Save misclassified sample details to CSV.

    Args:
        df: Misclassification dataframe.

    Returns:
        Path to the saved CSV file.
    """
    ensure_output_dirs()
    df.to_csv(MISCLASSIFIED_REPORT_PATH, index=False)
    logger.info("Saved misclassification report to %s", MISCLASSIFIED_REPORT_PATH)
    return MISCLASSIFIED_REPORT_PATH


def compute_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute counts for true-digit to predicted-digit confusion pairs.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dataframe sorted by most common incorrect confusion pairs.
    """
    incorrect_mask = y_true != y_pred
    pairs = pd.DataFrame(
        {
            "true_digit": y_true[incorrect_mask].astype(int),
            "predicted_digit": y_pred[incorrect_mask].astype(int),
        }
    )
    if pairs.empty:
        return pd.DataFrame(columns=["true_digit", "predicted_digit", "count"])

    return (
        pairs.value_counts(["true_digit", "predicted_digit"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def save_confusion_pairs(confusion_pairs: pd.DataFrame) -> Path:
    """Save confusion-pair counts to CSV.

    Args:
        confusion_pairs: Confusion-pair dataframe.

    Returns:
        Path to the saved CSV file.
    """
    ensure_output_dirs()
    confusion_pairs.to_csv(CONFUSION_PAIRS_PATH, index=False)
    logger.info("Saved confusion-pair report to %s", CONFUSION_PAIRS_PATH)
    return CONFUSION_PAIRS_PATH


def _save_prediction_grid(
    sample_indices: np.ndarray,
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_scores: np.ndarray,
    output_path: Path,
    title: str,
    max_examples: int = 25,
) -> Path:
    """Save a compact image grid for selected predictions."""
    ensure_output_dirs()
    selected_indices = sample_indices[:max_examples]
    if selected_indices.size == 0:
        logger.warning("No samples available for %s", output_path)
        return output_path

    grid_size = int(np.ceil(np.sqrt(len(selected_indices))))
    plt.figure(figsize=(11, 11))
    for plot_index, sample_index in enumerate(selected_indices, start=1):
        plt.subplot(grid_size, grid_size, plot_index)
        plt.imshow(x_raw[sample_index], cmap="gray")
        plt.title(
            f"T:{int(y_true[sample_index])} P:{int(y_pred[sample_index])}\n"
            f"C:{confidence_scores[sample_index]:.3f}",
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved prediction grid to %s", output_path)
    return output_path


def save_misclassified_grid(
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_scores: np.ndarray,
    max_examples: int = 25,
) -> Path:
    """Save a grid of misclassified examples.

    Args:
        x_raw: Raw test images.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidence_scores: Maximum predicted probability per sample.
        max_examples: Maximum examples to include.

    Returns:
        Path to the saved image file.
    """
    indices = get_misclassified_indices(y_true, y_pred)
    order = np.argsort(confidence_scores[indices])[::-1]
    return _save_prediction_grid(
        indices[order],
        x_raw,
        y_true,
        y_pred,
        confidence_scores,
        MISCLASSIFIED_GRID_PATH,
        "Misclassified MNIST Examples",
        max_examples=max_examples,
    )


def save_correct_prediction_grid(
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_scores: np.ndarray,
    min_confidence: float = 0.99,
    max_examples: int = 25,
) -> Path:
    """Save a grid of high-confidence correct examples.

    Args:
        x_raw: Raw test images.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidence_scores: Maximum predicted probability per sample.
        min_confidence: Minimum confidence for inclusion.
        max_examples: Maximum examples to include.

    Returns:
        Path to the saved image file.
    """
    correct_indices = np.where((y_true == y_pred) & (confidence_scores >= min_confidence))[0]
    order = np.argsort(confidence_scores[correct_indices])[::-1]
    return _save_prediction_grid(
        correct_indices[order],
        x_raw,
        y_true,
        y_pred,
        confidence_scores,
        CORRECT_GRID_PATH,
        "High-Confidence Correct MNIST Predictions",
        max_examples=max_examples,
    )


def run_error_analysis() -> dict[str, int | float]:
    """Run error analysis and save all requested artifacts.

    Returns:
        Summary metrics for the generated error analysis.
    """
    ensure_output_dirs()
    model, x_test, y_true, x_raw = load_model_and_test_data()
    y_pred, probabilities, confidence_scores = generate_predictions(model, x_test)
    misclassified_df = build_misclassification_dataframe(
        y_true,
        y_pred,
        confidence_scores,
        probabilities,
    )
    save_misclassification_report(misclassified_df)
    confusion_pairs = compute_confusion_pairs(y_true, y_pred)
    save_confusion_pairs(confusion_pairs)
    save_misclassified_grid(x_raw, y_true, y_pred, confidence_scores)
    save_correct_prediction_grid(x_raw, y_true, y_pred, confidence_scores)

    total_errors = int(len(misclassified_df))
    return {
        "total_errors": total_errors,
        "error_rate": float(total_errors / len(y_true)),
    }


def main() -> None:
    """Run the error-analysis workflow."""
    summary = run_error_analysis()
    logger.info(
        "Error analysis complete. Total errors: %s, error rate: %.4f",
        summary["total_errors"],
        summary["error_rate"],
    )


if __name__ == "__main__":
    main()
