"""Reusable Streamlit UI components for Phase 4B."""

import streamlit as st

from app.config import APP_NAME, APP_VERSION, MODEL_NAME
from app.ui.style import confidence_class


def render_hero() -> None:
    """Render compact premium hero section."""
    st.markdown(
        f"""
        <div class="studio-hero">
          <div class="badge-row">
            <span class="badge">CodeAlpha Machine Learning Internship Project</span>
            <span class="badge badge-success">Interactive Inference Ready</span>
          </div>
          <h1>Handwritten Digit Recognition Studio</h1>
          <p>CNN-powered handwritten digit recognition with preprocessing, confidence analysis, and saliency-based explainability.</p>
          <p>Built with TensorFlow • Keras • Streamlit • Explainable AI</p>
          <div class="badge-row">
            <span class="badge">CNN Model</span>
            <span class="badge">99.03% Test Accuracy</span>
            <span class="badge">MNIST Digits 0-9</span>
            <span class="badge badge-warning">v1.0.0 Ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render app sidebar and return selected page."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pages",
        ["Predict Digit", "Model Information", "Error Analysis", "About Project"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.subheader("Project")
    st.sidebar.write("CodeAlpha Handwritten Character Recognition")
    st.sidebar.caption("Educational and portfolio use only.")
    st.sidebar.success("Model Loaded | CNN Ready | 99.03% Test Accuracy | Prediction Ready")
    st.sidebar.subheader("Model")
    st.sidebar.write(MODEL_NAME)
    st.sidebar.write("TensorFlow / Keras")
    st.sidebar.write("28x28 grayscale")
    st.sidebar.write("Digits 0-9")
    st.sidebar.success("Interactive Inference Ready")
    st.sidebar.subheader("Progress")
    for item in [
        "[Complete] Data Pipeline",
        "[Complete] CNN Training",
        "[Complete] Evaluation",
        "[Complete] Explainability",
        "[Complete] Inference Services",
        "[Complete] Premium UI",
        "[Current] QA and Release",
    ]:
        st.sidebar.caption(item)
    st.sidebar.caption("v1.0.0 Release Candidate")
    return page


def section_header(title: str, caption: str | None = None) -> None:
    """Render a compact section header."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def metric_card(label: str, value: str) -> None:
    """Render a metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_badge(confidence_band: str) -> str:
    """Return styled confidence badge HTML."""
    css_class = confidence_class(confidence_band)
    return f'<span class="{css_class}">{confidence_band.title()}</span>'


def render_footer() -> None:
    """Render compact footer."""
    st.markdown(
        """
        <div class="footer">
        Developed by Syed Muzammil Shah<br/>
        TensorFlow • Keras • Streamlit • CodeAlpha Machine Learning Internship<br/>
        Educational and portfolio use only.
        </div>
        """,
        unsafe_allow_html=True,
    )
