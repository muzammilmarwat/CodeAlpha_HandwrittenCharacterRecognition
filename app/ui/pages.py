"""Streamlit page renderers for Phase 4B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import FINAL_REPORT_PATH, MODEL_CARD_PATH, MODEL_NAME
from app.services.model_service import get_model_metadata
from app.ui.charts import confusion_pair_chart, per_class_performance_chart
from app.ui.components import metric_card, section_header
from app.ui.downloads import render_download_center
from app.ui.prediction_workspace import render_prediction_workspace
from src.utils.paths import IMAGES_DIR, REPORTS_DIR


def _read_json(path: Path) -> dict:
    """Read a JSON report."""
    return json.loads(path.read_text(encoding="utf-8"))


def _display_image_if_exists(path: Path, caption: str) -> None:
    """Display image if it exists, otherwise show a missing-file state."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing image artifact: {path.name}")


def render_prediction_page() -> None:
    """Render prediction page."""
    render_prediction_workspace()


def render_model_information_page() -> None:
    """Render model information page."""
    section_header("Model Information", "Saved model metrics and training artifacts.")
    training_summary = _read_json(REPORTS_DIR / "training_summary.json")
    evaluation_summary = _read_json(REPORTS_DIR / "evaluation_summary.json")
    final_summary = pd.read_csv(REPORTS_DIR / "final_model_selection" / "final_model_selection_summary.csv").iloc[0]
    metadata = get_model_metadata()

    metric_cols = st.columns(5)
    metric_values = [
        ("Training Accuracy", f"{training_summary['final_training_accuracy']:.2%}"),
        ("Validation Accuracy", f"{training_summary['final_validation_accuracy']:.2%}"),
        ("Test Accuracy", f"{evaluation_summary['test_accuracy']:.2%}"),
        ("Test Loss", f"{evaluation_summary['test_loss']:.4f}"),
        ("Test Error Rate", f"{final_summary['test_error_rate']:.2%}"),
    ]
    for col, (label, value) in zip(metric_cols, metric_values):
        with col:
            metric_card(label, value)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Model Metadata")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Property": "Model", "Value": MODEL_NAME},
                    {"Property": "Framework", "Value": "TensorFlow / Keras"},
                    {"Property": "Input shape", "Value": str(metadata["input_shape"])},
                    {"Property": "Output shape", "Value": str(metadata["output_shape"])},
                    {"Property": "Optimizer", "Value": "Adam"},
                    {"Property": "Loss", "Value": "categorical_crossentropy"},
                    {"Property": "Batch size", "Value": training_summary["batch_size"]},
                    {"Property": "Epochs", "Value": training_summary["epochs_completed"]},
                    {"Property": "Random seed", "Value": training_summary.get("random_seed", "Not recorded")},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with col_right:
        st.subheader("Preprocessing Summary")
        st.write(
            "Uploaded images are converted to grayscale, polarity-corrected to match MNIST, "
            "cropped to foreground, resized with aspect ratio preserved, centered on a 28x28 canvas, "
            "normalized, and reshaped to a CNN tensor."
        )
        if MODEL_CARD_PATH.exists():
            st.download_button("Download Model Card", MODEL_CARD_PATH.read_bytes(), file_name="model_card.md")
        if FINAL_REPORT_PATH.exists():
            st.download_button(
                "Download Final Model Report",
                FINAL_REPORT_PATH.read_bytes(),
                file_name="final_model_selection_report.md",
            )

    section_header("Training and Evaluation Artifacts")
    grid = st.columns(2)
    images = [
        (IMAGES_DIR / "evaluation" / "training_accuracy.png", "Training Accuracy"),
        (IMAGES_DIR / "evaluation" / "training_loss.png", "Training Loss"),
        (IMAGES_DIR / "evaluation" / "confusion_matrix.png", "Confusion Matrix"),
        (IMAGES_DIR / "evaluation" / "sample_predictions.png", "Sample Predictions"),
        (IMAGES_DIR / "explainability" / "saliency_examples.png", "Saliency Examples"),
    ]
    for index, (path, caption) in enumerate(images):
        with grid[index % 2]:
            _display_image_if_exists(path, caption)


def render_error_analysis_page() -> None:
    """Render error analysis page."""
    section_header("Error Analysis", "Review confidence, class-level behavior, and common confusions.")
    final_summary = pd.read_csv(REPORTS_DIR / "final_model_selection" / "final_model_selection_summary.csv").iloc[0]
    confidence_summary = _read_json(REPORTS_DIR / "explainability" / "confidence_summary.json")
    per_class_df = pd.read_csv(REPORTS_DIR / "explainability" / "per_class_performance.csv")
    confusion_pairs = pd.read_csv(REPORTS_DIR / "error_analysis" / "top_confusion_pairs.csv")

    metric_cols = st.columns(5)
    metrics = [
        ("Total Errors", str(int(final_summary["total_test_errors"]))),
        ("Error Rate", f"{final_summary['test_error_rate']:.2%}"),
        ("Strongest Digit", str(int(final_summary["strongest_digit_class"]))),
        ("Weakest Digit", str(int(final_summary["weakest_digit_class"]))),
        ("Incorrect Confidence", f"{confidence_summary['mean_confidence_incorrect']:.2%}"),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            metric_card(label, value)

    st.write(
        "Most mistakes are visually plausible confusions. Digits such as 4 and 9, "
        "2 and 7, 3 and 5, and 9 and 8 can share strokes or shapes after resizing."
    )
    col_pairs, col_class = st.columns(2)
    with col_pairs:
        st.subheader("Common Confusion Pairs")
        confusion_pair_chart(confusion_pairs)
        st.dataframe(confusion_pairs.head(10), hide_index=True, use_container_width=True)
    with col_class:
        st.subheader("Per-Class Accuracy")
        per_class_performance_chart(per_class_df)
        st.dataframe(per_class_df, hide_index=True, use_container_width=True)

    section_header("Analysis Images")
    image_cols = st.columns(2)
    images = [
        (IMAGES_DIR / "error_analysis" / "misclassified_examples.png", "Misclassified Examples"),
        (IMAGES_DIR / "error_analysis" / "high_confidence_correct_predictions.png", "High-Confidence Correct Predictions"),
        (IMAGES_DIR / "explainability" / "confidence_distribution.png", "Confidence Distribution"),
        (IMAGES_DIR / "explainability" / "confidence_correct_vs_incorrect.png", "Correct vs Incorrect Confidence"),
        (IMAGES_DIR / "explainability" / "per_class_accuracy.png", "Per-Class Accuracy"),
    ]
    for index, (path, caption) in enumerate(images):
        with image_cols[index % 2]:
            _display_image_if_exists(path, caption)


def render_about_page() -> None:
    """Render about page."""
    section_header("About Project", "A complete CNN-based handwritten digit recognition workflow.")
    st.write("Project: CodeAlpha Handwritten Character Recognition")
    st.write("Objective: Recognize handwritten digits using a CNN trained on MNIST.")

    col_dataset, col_stack = st.columns(2)
    with col_dataset:
        st.subheader("Dataset")
        st.write("- MNIST")
        st.write("- 60,000 training images")
        st.write("- 10,000 test images")
        st.write("- 28x28 grayscale")
        st.write("- 10 classes")
    with col_stack:
        st.subheader("Tech Stack")
        st.write("Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit, Pytest")

    st.subheader("Workflow")
    st.code(
        "MNIST Dataset -> EDA -> Preprocessing -> CNN Training -> Evaluation -> "
        "Error Analysis -> Confidence Analysis -> Explainability -> Saved Model -> Streamlit Inference",
        language="text",
    )

    st.subheader("Limitations")
    st.write("- MNIST is clean compared with real-world handwriting.")
    st.write("- The model supports digits only, not alphabet recognition.")
    st.write("- Saliency is not causal explanation.")
    st.write("- No adversarial robustness testing has been performed.")
    st.write("- Educational and portfolio use only.")

    render_download_center(None, st.session_state.get("prediction_history", []))

