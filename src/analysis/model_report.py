"""Generate model selection, model card, and saliency artifacts."""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

from src.analysis.confidence_analysis import (
    CONFIDENCE_SUMMARY_PATH,
    PER_CLASS_PERFORMANCE_PATH,
    run_confidence_analysis,
)
from src.analysis.error_analysis import (
    CONFUSION_PAIRS_PATH,
    MISCLASSIFIED_REPORT_PATH,
    generate_predictions,
    load_model_and_test_data,
    run_error_analysis,
)
from src.utils.logging_config import setup_logger
from src.utils.paths import IMAGES_DIR, MODELS_DIR, REPORTS_DIR


logger = setup_logger(__name__)

FINAL_SELECTION_DIR = REPORTS_DIR / "final_model_selection"
EXPLAINABILITY_IMAGES_DIR = IMAGES_DIR / "explainability"
FINAL_MODEL_REPORT_PATH = FINAL_SELECTION_DIR / "final_model_selection_report.md"
FINAL_MODEL_SUMMARY_PATH = FINAL_SELECTION_DIR / "final_model_selection_summary.csv"
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"
SALIENCY_EXAMPLES_PATH = EXPLAINABILITY_IMAGES_DIR / "saliency_examples.png"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"


def ensure_output_dirs() -> None:
    """Create output directories for report and explainability artifacts."""
    FINAL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file with a defensive file check."""
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_analysis_artifacts() -> None:
    """Create Phase 3 analysis artifacts when they are not already present."""
    if not MISCLASSIFIED_REPORT_PATH.exists() or not CONFUSION_PAIRS_PATH.exists():
        logger.info("Error-analysis reports missing; generating them now")
        run_error_analysis()
    if not CONFIDENCE_SUMMARY_PATH.exists() or not PER_CLASS_PERFORMANCE_PATH.exists():
        logger.info("Confidence-analysis reports missing; generating them now")
        run_confidence_analysis()


def _format_percentage(value: float) -> str:
    """Format a decimal metric as a percentage string."""
    return f"{value * 100:.2f}%"


def _load_phase_metrics() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load saved training and evaluation summaries."""
    return _read_json(TRAINING_SUMMARY_PATH), _read_json(EVALUATION_SUMMARY_PATH)


def _get_class_extremes(per_class_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify strongest and weakest digit classes."""
    strongest_row = per_class_df.sort_values(
        ["f1_score", "recall"],
        ascending=False,
    ).iloc[0]
    weakest_row = per_class_df.sort_values(["f1_score", "recall"]).iloc[0]
    lowest_recall_row = per_class_df.sort_values("recall").iloc[0]
    return {
        "strongest_digit": int(strongest_row["digit"]),
        "strongest_f1": float(strongest_row["f1_score"]),
        "weakest_digit": int(weakest_row["digit"]),
        "weakest_f1": float(weakest_row["f1_score"]),
        "lowest_recall_digit": int(lowest_recall_row["digit"]),
        "lowest_recall": float(lowest_recall_row["recall"]),
    }


def create_saliency_map(
    model: Model,
    image: np.ndarray,
    predicted_class: int,
) -> np.ndarray:
    """Create a lightweight gradient-based saliency map.

    Saliency maps indicate pixel sensitivity, not causal explanation.

    Args:
        model: Trained Keras model.
        image: Single processed image shaped as (28, 28, 1).
        predicted_class: Class index used as the gradient target.

    Returns:
        Normalized saliency map shaped as (28, 28).
    """
    image_tensor = tf.convert_to_tensor(np.expand_dims(image.astype("float32"), axis=0))
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, image_tensor)
    if gradients is None:
        return np.zeros(image.shape[:2], dtype="float32")

    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0].numpy()
    min_value = float(np.min(saliency))
    max_value = float(np.max(saliency))
    if np.isclose(max_value, min_value):
        return np.zeros_like(saliency, dtype="float32")

    return ((saliency - min_value) / (max_value - min_value)).astype("float32")


def save_saliency_examples(
    model: Model,
    x_processed: np.ndarray,
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    max_examples: int = 5,
) -> Path:
    """Save saliency maps for correctly predicted high-confidence samples.

    Args:
        model: Trained Keras model.
        x_processed: Processed test images.
        x_raw: Raw test images.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Full class probability matrix.
        max_examples: Maximum number of examples to show.

    Returns:
        Path to the saved saliency image.
    """
    ensure_output_dirs()
    confidence_scores = np.max(probabilities, axis=1)
    correct_indices = np.where(y_true == y_pred)[0]
    ordered_indices = correct_indices[np.argsort(confidence_scores[correct_indices])[::-1]]
    selected_indices = ordered_indices[:max_examples]

    if selected_indices.size == 0:
        logger.warning("No correct samples available for saliency examples")
        return SALIENCY_EXAMPLES_PATH

    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(6, 2.8 * len(selected_indices)))
    axes_array = np.atleast_2d(axes)
    for row_index, sample_index in enumerate(selected_indices):
        predicted_class = int(y_pred[sample_index])
        saliency = create_saliency_map(model, x_processed[sample_index], predicted_class)
        confidence = float(confidence_scores[sample_index])

        axes_array[row_index, 0].imshow(x_raw[sample_index], cmap="gray")
        axes_array[row_index, 0].set_title(
            f"Digit {int(y_true[sample_index])} | Pred {predicted_class}\n"
            f"Confidence {confidence:.3f}",
            fontsize=9,
        )
        axes_array[row_index, 0].axis("off")

        axes_array[row_index, 1].imshow(saliency, cmap="magma")
        axes_array[row_index, 1].set_title("Saliency heatmap", fontsize=9)
        axes_array[row_index, 1].axis("off")

    fig.suptitle("Saliency maps show pixel sensitivity, not causal explanation.", fontsize=11)
    plt.tight_layout()
    plt.savefig(SALIENCY_EXAMPLES_PATH, dpi=150)
    plt.close()
    logger.info("Saved saliency examples to %s", SALIENCY_EXAMPLES_PATH)
    return SALIENCY_EXAMPLES_PATH


def _model_architecture_lines(model: Model) -> list[str]:
    """Return concise layer descriptions from the loaded model."""
    lines = []
    for layer in model.layers:
        class_name = layer.__class__.__name__
        config = layer.get_config()
        if class_name in {"Conv2D", "Dense"}:
            lines.append(
                f"- {class_name}: units/filters={config.get('units', config.get('filters'))}, "
                f"activation={config.get('activation')}"
            )
        elif class_name in {"MaxPooling2D", "Dropout", "Flatten"}:
            detail = config.get("rate", config.get("pool_size", ""))
            lines.append(f"- {class_name}: {detail}")
    return lines


def _top_confusion_lines(confusion_pairs: pd.DataFrame, limit: int = 5) -> list[str]:
    """Format the most common confusion pairs as markdown bullets."""
    if confusion_pairs.empty:
        return ["- No incorrect confusion pairs found."]
    return [
        f"- True {int(row.true_digit)} -> Predicted {int(row.predicted_digit)}: {int(row.count)}"
        for row in confusion_pairs.head(limit).itertuples(index=False)
    ]


def _write_final_model_report(
    training_summary: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    confidence_summary: Dict[str, Any],
    per_class_df: pd.DataFrame,
    confusion_pairs: pd.DataFrame,
    misclassified_df: pd.DataFrame,
) -> None:
    """Write the final model selection markdown report."""
    extremes = _get_class_extremes(per_class_df)
    total_errors = int(len(misclassified_df))
    test_samples = int(evaluation_summary["test_samples"])
    error_rate = total_errors / test_samples
    top_confusions = "\n".join(_top_confusion_lines(confusion_pairs))

    report = f"""# Final Model Selection Report

## Executive Summary
The `mnist_cnn_baseline` model is selected as the final model for the MNIST
handwritten digit recognition workflow.

## Model Performance
- Training accuracy: {_format_percentage(training_summary["final_training_accuracy"])}
- Validation accuracy: {_format_percentage(training_summary["final_validation_accuracy"])}
- Best validation accuracy: {_format_percentage(training_summary["best_validation_accuracy"])}
- Test accuracy: {_format_percentage(evaluation_summary["test_accuracy"])}
- Test loss: {evaluation_summary["test_loss"]:.4f}

## Per-Class Performance
Strongest class by F1-score: digit {extremes["strongest_digit"]} with F1
{extremes["strongest_f1"]:.4f}. Weakest class by F1-score: digit
{extremes["weakest_digit"]} with F1 {extremes["weakest_f1"]:.4f}. Lowest recall
was observed for digit {extremes["lowest_recall_digit"]} with recall
{extremes["lowest_recall"]:.4f}.

## Error Analysis
- Total incorrect predictions: {total_errors}
- Test error rate: {_format_percentage(error_rate)}
- Average confidence of incorrect predictions: {confidence_summary["mean_confidence_incorrect"]:.4f}

Most common confusion pairs:
{top_confusions}

## Confidence Analysis
Correct predictions have mean confidence
{confidence_summary["mean_confidence_correct"]:.4f}, while incorrect
predictions have mean confidence {confidence_summary["mean_confidence_incorrect"]:.4f}.
Low-confidence predictions should be surfaced clearly in deployment.

## Explainability
Gradient-based saliency maps were generated for selected correct predictions.
These maps show pixel sensitivity for the predicted class score, not causal
explanation.

## Final Selection Rationale
The baseline CNN is selected because it delivers strong test accuracy,
consistent validation and test performance, a low error rate, and a lightweight
architecture suitable for a deployment demonstration.

## Limitations
- MNIST is clean and standardized.
- Real handwriting may be noisier.
- This model supports digits only, not the full alphabet.
- Saliency maps are not causal explanations.
- No adversarial robustness testing has been performed.
- No calibration analysis has been completed yet.

## Deployment Recommendation
Use `mnist_cnn_baseline.keras`, apply the same preprocessing used during
training, validate image dimensions and polarity, display prediction
probabilities, and show confidence warnings for low-confidence predictions.
"""
    FINAL_MODEL_REPORT_PATH.write_text(report, encoding="utf-8")


def _write_final_model_summary(
    training_summary: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    confidence_summary: Dict[str, Any],
    per_class_df: pd.DataFrame,
    misclassified_df: pd.DataFrame,
) -> None:
    """Write a compact CSV summary for final model selection."""
    extremes = _get_class_extremes(per_class_df)
    summary = pd.DataFrame(
        [
            {
                "selected_model": "mnist_cnn_baseline",
                "training_accuracy": training_summary["final_training_accuracy"],
                "validation_accuracy": training_summary["final_validation_accuracy"],
                "test_accuracy": evaluation_summary["test_accuracy"],
                "test_loss": evaluation_summary["test_loss"],
                "total_test_errors": len(misclassified_df),
                "test_error_rate": len(misclassified_df) / evaluation_summary["test_samples"],
                "strongest_digit_class": extremes["strongest_digit"],
                "weakest_digit_class": extremes["weakest_digit"],
                "mean_confidence_correct": confidence_summary["mean_confidence_correct"],
                "mean_confidence_incorrect": confidence_summary["mean_confidence_incorrect"],
            }
        ]
    )
    summary.to_csv(FINAL_MODEL_SUMMARY_PATH, index=False)


def _write_model_card(
    model: Model,
    training_summary: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    per_class_df: pd.DataFrame,
) -> None:
    """Write the project model card."""
    extremes = _get_class_extremes(per_class_df)
    architecture = "\n".join(_model_architecture_lines(model))
    top_classes = per_class_df.sort_values("f1_score", ascending=False).head(3)
    weak_classes = per_class_df.sort_values("f1_score").head(3)

    model_card = f"""# Model Card: MNIST CNN Digit Classifier

## Model Overview
- Model name: `mnist_cnn_baseline`
- Architecture type: Convolutional Neural Network
- Task: Handwritten digit classification
- Framework: TensorFlow/Keras
- Input shape: 28 x 28 x 1
- Output classes: digits 0-9
- Status: Selected final baseline model

## Intended Use
Educational, internship, and portfolio use for demonstrating a complete
computer vision workflow.

## Dataset
- MNIST
- 60,000 training images
- 10,000 test images
- Digits 0-9
- Grayscale 28 x 28 images

## Preprocessing
- Pixel normalization to [0, 1]
- Reshape to `(28, 28, 1)`
- One-hot label encoding during training

## Architecture
{architecture}

## Evaluation Metrics
- Training accuracy: {_format_percentage(training_summary["final_training_accuracy"])}
- Validation accuracy: {_format_percentage(training_summary["final_validation_accuracy"])}
- Test accuracy: {_format_percentage(evaluation_summary["test_accuracy"])}
- Test loss: {evaluation_summary["test_loss"]:.4f}

## Per-Class Results
Strongest digit class: {extremes["strongest_digit"]}. Weakest digit class:
{extremes["weakest_digit"]}.

Top classes by F1-score:
{top_classes[["digit", "f1_score", "recall"]].to_markdown(index=False)}

Weakest classes by F1-score:
{weak_classes[["digit", "f1_score", "recall"]].to_markdown(index=False)}

## Known Limitations
- MNIST images are cleaner and more standardized than real handwriting.
- The classifier supports digits only.
- Input polarity and centering must match the training distribution.
- Saliency maps indicate sensitivity, not causal explanation.

## Ethical and Safety Notes
This model is suitable for low-risk educational demonstrations. It should not
be used for high-stakes identity, grading, or accessibility decisions without
additional validation, monitoring, and human review.

## Future Improvements
- EMNIST support
- Data augmentation
- Confidence calibration
- Robustness testing
- Handwriting canvas
- Deployment monitoring
"""
    MODEL_CARD_PATH.write_text(model_card, encoding="utf-8")


def _ensure_markdown_support() -> None:
    """Patch pandas markdown output fallback when tabulate is unavailable."""
    try:
        pd.DataFrame({"a": [1]}).to_markdown(index=False)
    except ImportError:
        pd.DataFrame.to_markdown = _dataframe_to_markdown  # type: ignore[method-assign]


def _dataframe_to_markdown(self: pd.DataFrame, index: bool = False) -> str:
    """Small markdown fallback for compact model-card tables."""
    data = self.reset_index() if index else self.copy()
    headers = [str(column) for column in data.columns]
    rows = data.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_model_reports() -> Dict[str, Any]:
    """Generate final model selection reports, model card, and saliency image.

    Returns:
        Summary of key model-selection values.
    """
    ensure_output_dirs()
    _ensure_markdown_support()
    _ensure_analysis_artifacts()

    training_summary, evaluation_summary = _load_phase_metrics()
    confidence_summary = _read_json(CONFIDENCE_SUMMARY_PATH)
    per_class_df = pd.read_csv(PER_CLASS_PERFORMANCE_PATH)
    confusion_pairs = pd.read_csv(CONFUSION_PAIRS_PATH)
    misclassified_df = pd.read_csv(MISCLASSIFIED_REPORT_PATH)

    model, x_processed, y_true, x_raw = load_model_and_test_data()
    y_pred, probabilities, _ = generate_predictions(model, x_processed)
    save_saliency_examples(model, x_processed, x_raw, y_true, y_pred, probabilities)

    _write_final_model_report(
        training_summary,
        evaluation_summary,
        confidence_summary,
        per_class_df,
        confusion_pairs,
        misclassified_df,
    )
    _write_final_model_summary(
        training_summary,
        evaluation_summary,
        confidence_summary,
        per_class_df,
        misclassified_df,
    )
    _write_model_card(model, training_summary, evaluation_summary, per_class_df)

    extremes = _get_class_extremes(per_class_df)
    summary = {
        "selected_model": "mnist_cnn_baseline",
        "total_test_errors": int(len(misclassified_df)),
        "error_rate": float(len(misclassified_df) / evaluation_summary["test_samples"]),
        "strongest_digit_class": extremes["strongest_digit"],
        "weakest_digit_class": extremes["weakest_digit"],
        "mean_confidence_correct": confidence_summary["mean_confidence_correct"],
        "mean_confidence_incorrect": confidence_summary["mean_confidence_incorrect"],
    }
    logger.info("Generated final model reports and model card")
    return summary


def main() -> None:
    """Run final model report generation."""
    summary = generate_model_reports()
    logger.info(
        "Final model: %s; errors: %s; error rate: %.4f",
        summary["selected_model"],
        summary["total_test_errors"],
        summary["error_rate"],
    )


if __name__ == "__main__":
    main()
