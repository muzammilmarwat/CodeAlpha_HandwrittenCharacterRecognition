"""Prediction workspace for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.config import DEFAULT_CANVAS_SIZE, MAX_UPLOAD_SIZE_MB, SUPPORTED_IMAGE_TYPES
from app.schemas.prediction_schema import PredictionResult
from app.services.history_service import append_history, clear_history, create_history_record
from app.services.prediction_service import predict_from_canvas, predict_from_uploaded_image
from app.services.sample_service import get_sample_digit, predict_sample_digit
from app.ui.charts import all_class_probability_chart, confidence_visual, top_3_probability_chart
from app.ui.components import metric_card, section_header
from app.ui.downloads import render_download_center
from app.utils.exceptions import AppError, CanvasNotAvailableError
from app.utils.logging_config import get_logger


logger = get_logger(__name__)


def _load_canvas_component():
    """Load optional canvas component if available."""
    try:
        from streamlit_drawable_canvas import st_canvas

        return st_canvas
    except Exception:
        return None


def initialize_prediction_state() -> None:
    """Initialize prediction-related session state."""
    defaults = {
        "prediction_history": [],
        "latest_prediction": None,
        "latest_input_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _store_prediction(result: PredictionResult) -> None:
    """Persist latest result and append history."""
    st.session_state.latest_prediction = result
    st.session_state.prediction_history = append_history(
        st.session_state.prediction_history,
        create_history_record(result),
    )


def render_input_guidance() -> None:
    """Render compact input recommendations."""
    st.info(
        "Use one centered handwritten digit with high contrast and minimal background noise. "
        f"Supported formats: {', '.join(SUPPORTED_IMAGE_TYPES).upper()}. "
        f"Maximum upload size: {MAX_UPLOAD_SIZE_MB} MB."
    )


def render_upload_source() -> None:
    """Render upload input and prediction action."""
    render_input_guidance()
    uploaded_file = st.file_uploader(
        "Upload image",
        type=list(SUPPORTED_IMAGE_TYPES),
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.caption("No image selected yet. Upload a digit image to begin.")
        return

    st.caption(f"Selected: {uploaded_file.name}")
    if st.button("Recognize Digit", type="primary", use_container_width=True, key="predict_upload"):
        with st.spinner("Preprocessing image and running CNN inference..."):
            try:
                result = predict_from_uploaded_image(uploaded_file)
                _store_prediction(result)
            except AppError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Unexpected upload prediction failure")
                st.error("Unexpected prediction failure. Please try another image.")


def render_sample_source() -> None:
    """Render sample digit gallery and prediction action."""
    digit = st.select_slider("Choose example digit", options=list(range(10)), value=7)
    image, true_digit = get_sample_digit(digit=int(digit))
    col_preview, col_action = st.columns([1, 2])
    with col_preview:
        st.caption(f"MNIST sample digit {true_digit}")
        st.image(image.resize((140, 140)), width=140)
    with col_action:
        st.write("Example digits are loaded dynamically from the MNIST test set.")
        if st.button("Recognize Digit", type="primary", use_container_width=True, key="predict_sample"):
            with st.spinner("Running sample inference..."):
                try:
                    result = predict_sample_digit(digit=int(digit))
                    _store_prediction(result)
                except AppError as exc:
                    st.error(str(exc))


def render_canvas_source() -> None:
    """Render optional drawing canvas or disabled state."""
    st_canvas = _load_canvas_component()
    if st_canvas is None:
        st.info(
            "Drawing canvas is planned. Upload and sample inference are fully supported. "
            "The canvas package was not enabled because it has not been verified in this environment."
        )
        return

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=DEFAULT_CANVAS_SIZE,
        height=DEFAULT_CANVAS_SIZE,
        drawing_mode="freedraw",
        key="digit_canvas",
    )
    if st.button("Recognize Digit", type="primary", use_container_width=True, key="predict_canvas"):
        if canvas_result.image_data is None:
            raise CanvasNotAvailableError("Canvas has no image data.")
        try:
            result = predict_from_canvas(np.asarray(canvas_result.image_data).astype("uint8"))
            _store_prediction(result)
        except AppError as exc:
            st.error(str(exc))


def render_preprocessing_workspace(result: PredictionResult) -> None:
    """Render original/processed/saliency visual workspace."""
    section_header("Preprocessing Workspace", "Exact model input preview generated by the app pipeline.")
    col_original, col_processed, col_saliency = st.columns(3)
    with col_original:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.caption("Original Image")
        st.image(result.original_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_processed:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.caption("Processed 28x28 Image")
        st.image(result.processed_image, clamp=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_saliency:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.caption("Saliency Heatmap")
        if result.saliency_map is not None:
            st.image(result.saliency_map, clamp=True, use_container_width=True)
        else:
            st.warning("Saliency unavailable for this prediction.")
        st.markdown("</div>", unsafe_allow_html=True)

    pipeline = "Input -> Grayscale -> Polarity Correction -> Crop -> Resize -> Center -> Normalize -> CNN Tensor"
    st.caption(pipeline)
    properties = pd.DataFrame(
        [
            {"Property": "Source type", "Value": result.source_type},
            {"Property": "Processed dimensions", "Value": "28 x 28"},
            {"Property": "Tensor shape", "Value": "(1, 28, 28, 1)"},
            {"Property": "Model", "Value": result.model_name},
        ]
    )
    st.dataframe(properties, hide_index=True, use_container_width=True)


def render_result_dashboard(result: PredictionResult) -> None:
    """Render polished prediction dashboard."""
    section_header("Recognition Result")
    col_digit, col_confidence, col_margin = st.columns([1.1, 1, 1])
    with col_digit:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.caption("Predicted Digit")
        st.markdown(f'<div class="digit-result">{result.predicted_digit}</div>', unsafe_allow_html=True)
        st.caption(f"Source: {result.source_type.title()} | Model: MNIST CNN Baseline")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_confidence:
        metric_card("Confidence", f"{result.confidence:.2%}")
        metric_card("Confidence Band", result.confidence_band.title())
    with col_margin:
        confidence_visual(result)

    if result.warning_message:
        st.warning(result.warning_message)
    else:
        st.success(
            f"The model is highly confident that this image represents the digit {result.predicted_digit}."
        )

    section_header("Probability Dashboard")
    col_top, col_all = st.columns([1, 1])
    with col_top:
        st.write("Top 3 Predictions")
        top_3_probability_chart(result)
        top_table = pd.DataFrame(
            [
                {
                    "Rank": item.rank,
                    "Digit": item.digit,
                    "Probability": f"{item.probability:.2%}",
                }
                for item in result.top_predictions
            ]
        )
        st.dataframe(top_table, hide_index=True, use_container_width=True)
    with col_all:
        with st.expander("View all digit probabilities", expanded=True):
            all_class_probability_chart(result)

    section_header("Explainability")
    st.caption("Highlighted pixels had the strongest influence on the model's prediction.")
    st.caption("Saliency indicates model sensitivity. It is not a causal explanation.")


def render_history_panel() -> None:
    """Render prediction history and history downloads."""
    section_header("Prediction History")
    history = st.session_state.prediction_history
    if not history:
        st.info("Prediction history is empty. Run a prediction to populate this session table.")
        return
    history_df = pd.DataFrame(
        [
            {
                "Time": record.timestamp.strftime("%H:%M:%S"),
                "Source": record.source_type,
                "Predicted Digit": record.predicted_digit,
                "Confidence": f"{record.confidence:.2%}",
                "Confidence Band": record.confidence_band,
                "Top-3 Summary": record.top_3_summary,
            }
            for record in history
        ]
    )
    st.dataframe(history_df, hide_index=True, use_container_width=True)
    if st.button("Clear History", use_container_width=True):
        st.session_state.prediction_history = clear_history()
        st.rerun()


def render_prediction_workspace() -> None:
    """Render the main prediction page workspace."""
    initialize_prediction_state()
    section_header("Input Source", "Choose an upload, sample digit, or canvas when available.")
    input_source = st.radio(
        "Input source",
        ["Upload Image", "Example Digit", "Draw Digit"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if input_source == "Upload Image":
        render_upload_source()
    elif input_source == "Example Digit":
        render_sample_source()
    else:
        render_canvas_source()

    result = st.session_state.latest_prediction
    if result is None:
        st.info("No prediction yet. Select an input and click Recognize Digit.")
    else:
        render_preprocessing_workspace(result)
        render_result_dashboard(result)

    render_history_panel()
    render_download_center(st.session_state.latest_prediction, st.session_state.prediction_history)

