"""Evaluate the trained MNIST CNN model."""

import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from src.data.data_loader import get_class_names, load_mnist_data
from src.preprocessing.image_preprocessor import preprocess_mnist_data
from src.utils.logging_config import setup_logger
from src.utils.paths import IMAGES_DIR, MODELS_DIR, REPORTS_DIR


logger = setup_logger(__name__)

MODEL_PATH = MODELS_DIR / "mnist_cnn_baseline.keras"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_IMAGE_PATH = IMAGES_DIR / "evaluation" / "confusion_matrix.png"
SAMPLE_PREDICTIONS_PATH = IMAGES_DIR / "evaluation" / "sample_predictions.png"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.json"


def ensure_output_dirs() -> None:
    """Create output directories required by the evaluation workflow."""
    for directory in [REPORTS_DIR, IMAGES_DIR / "evaluation"]:
        directory.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_plot(matrix: np.ndarray, class_labels: list[str]) -> None:
    """Save a confusion matrix heatmap.

    Args:
        matrix: Confusion matrix values.
        class_labels: Label names for axes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMAGE_PATH, dpi=150)
    plt.close()


def save_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: str | None = None,
    sample_count: int = 25,
) -> None:
    """Save a grid of sample MNIST predictions.

    Args:
        images: Raw image array.
        true_labels: Ground-truth labels.
        predicted_labels: Predicted labels.
        output_path: Optional image destination.
        sample_count: Number of samples to display.
    """
    destination = SAMPLE_PREDICTIONS_PATH if output_path is None else output_path
    grid_size = int(np.ceil(np.sqrt(sample_count)))
    plt.figure(figsize=(10, 10))

    for index in range(sample_count):
        plt.subplot(grid_size, grid_size, index + 1)
        plt.imshow(images[index], cmap="gray")
        title_color = "green" if true_labels[index] == predicted_labels[index] else "red"
        plt.title(
            f"T:{true_labels[index]} P:{predicted_labels[index]}",
            color=title_color,
            fontsize=9,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def evaluate_trained_model() -> Dict[str, Any]:
    """Evaluate the saved baseline CNN on the MNIST test set.

    Returns:
        Evaluation summary dictionary.

    Raises:
        FileNotFoundError: If the trained model artifact does not exist.
    """
    ensure_output_dirs()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run training first."
        )

    (_, _), (x_test, y_test) = load_mnist_data()
    _, _, x_test_processed, y_test_processed = preprocess_mnist_data(
        x_test,
        y_test,
        x_test,
        y_test,
    )

    logger.info("Loading trained model from %s", MODEL_PATH)
    model = load_model(MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(
        x_test_processed,
        y_test_processed,
        verbose=0,
    )

    probabilities = model.predict(x_test_processed, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    class_names = get_class_names()
    class_labels = list(class_names.values())

    report = classification_report(
        y_test,
        predicted_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_PATH)

    matrix = confusion_matrix(y_test, predicted_labels)
    pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_csv(
        CONFUSION_MATRIX_CSV_PATH
    )
    save_confusion_matrix_plot(matrix, class_labels)
    save_sample_predictions(x_test, y_test, predicted_labels)

    summary: Dict[str, Any] = {
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_samples": int(x_test.shape[0]),
    }
    EVALUATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Evaluation complete. Test accuracy: %.4f, test loss: %.4f",
        test_accuracy,
        test_loss,
    )
    return summary


def main() -> None:
    """Run the trained model evaluation workflow."""
    evaluate_trained_model()


if __name__ == "__main__":
    main()
