"""Premium Streamlit app router for handwritten digit recognition."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import APP_NAME
from app.ui.components import render_footer, render_hero, render_sidebar
from app.ui.pages import (
    render_about_page,
    render_error_analysis_page,
    render_model_information_page,
    render_prediction_page,
)
from app.ui.prediction_workspace import initialize_prediction_state
from app.ui.style import apply_global_style
from app.utils.exceptions import AppError
from app.utils.logging_config import configure_app_logging
from app.utils.paths import validate_required_artifacts


def main() -> None:
    """Run the Streamlit application."""
    configure_app_logging()
    st.set_page_config(
        page_title="Handwritten Digit Recognition Studio",
        page_icon="123",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_style()
    initialize_prediction_state()

    try:
        validate_required_artifacts()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    page = render_sidebar()
    render_hero()

    if page == "Predict Digit":
        render_prediction_page()
    elif page == "Model Information":
        render_model_information_page()
    elif page == "Error Analysis":
        render_error_analysis_page()
    else:
        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()
