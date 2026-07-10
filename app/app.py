"""Minimal Streamlit shell for handwritten digit inference."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app.config import APP_NAME, APP_VERSION, DEFAULT_CANVAS_SIZE, SUPPORTED_IMAGE_TYPES
from app.services.history_service import append_history, create_history_record
from app.services.prediction_service import predict_from_canvas, predict_from_uploaded_image
from app.services.sample_service import predict_sample_digit
from app.ui.minimal_components import render_history, render_prediction_result
from app.utils.exceptions import AppError, CanvasNotAvailableError
from app.utils.logging_config import configure_app_logging, get_logger
from app.utils.paths import validate_required_artifacts


logger = get_logger(__name__)


def _load_canvas_component():
    """Load optional canvas component if installed."""
    try:
        from streamlit_drawable_canvas import st_canvas

        return st_canvas
    except Exception:
        return None


def _initialize_session_state() -> None:
    """Initialize Streamlit session state values."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def _record_result(result) -> None:
    """Append a prediction result to session history."""
    record = create_history_record(result)
    st.session_state.prediction_history = append_history(st.session_state.prediction_history, record)


def render_upload_tab() -> None:
    """Render upload inference controls."""
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image",
        type=list(SUPPORTED_IMAGE_TYPES),
    )
    if uploaded_file is None:
        st.info("Upload a PNG, JPG, or JPEG image containing one handwritten digit.")
        return

    try:
        result = predict_from_uploaded_image(uploaded_file)
        _record_result(result)
        render_prediction_result(result)
    except AppError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected upload prediction failure")
        st.error("Unexpected prediction failure. Please try another image.")


def render_sample_tab() -> None:
    """Render dynamic MNIST sample controls."""
    digit = st.selectbox("Choose a sample digit", list(range(10)), index=0)
    if st.button("Predict sample digit"):
        try:
            result = predict_sample_digit(digit=int(digit))
            _record_result(result)
            render_prediction_result(result)
        except AppError as exc:
            st.error(str(exc))


def render_canvas_tab() -> None:
    """Render optional drawing canvas controls."""
    st_canvas = _load_canvas_component()
    if st_canvas is None:
        st.info(
            "Drawing canvas is not enabled in this environment. "
            "Image upload and sample digits are fully supported for Phase 4A."
        )
        return

    st.caption("Draw one white digit on the black canvas.")
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
    if st.button("Predict drawing"):
        if canvas_result.image_data is None:
            raise CanvasNotAvailableError("Canvas has no image data.")
        try:
            canvas_array = np.asarray(canvas_result.image_data).astype("uint8")
            result = predict_from_canvas(canvas_array)
            _record_result(result)
            render_prediction_result(result)
        except AppError as exc:
            st.error(str(exc))


def main() -> None:
    """Run the minimal Streamlit inference shell."""
    configure_app_logging()
    st.set_page_config(page_title=APP_NAME, page_icon="ML", layout="wide")
    _initialize_session_state()

    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION} | Phase 4A functional inference shell")

    try:
        validate_required_artifacts()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    upload_tab, sample_tab, canvas_tab = st.tabs(["Upload", "Sample Digit", "Canvas"])
    with upload_tab:
        render_upload_tab()
    with sample_tab:
        render_sample_tab()
    with canvas_tab:
        render_canvas_tab()

    render_history(st.session_state.prediction_history)


if __name__ == "__main__":
    main()

