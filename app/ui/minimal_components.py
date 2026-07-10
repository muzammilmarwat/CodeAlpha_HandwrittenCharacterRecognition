"""Small Streamlit rendering helpers for Phase 4A."""

import pandas as pd
import streamlit as st

from app.schemas.prediction_schema import HistoryRecord, PredictionResult
from app.services.image_service import create_preprocessing_preview
from app.services.saliency_service import SALIENCY_DISCLAIMER


def render_prediction_result(result: PredictionResult) -> None:
    """Render a functional prediction result block.

    Args:
        result: Prediction result to display.
    """
    st.subheader(f"Predicted Digit: {result.predicted_digit}")
    st.metric("Confidence", f"{result.confidence:.2%}", result.confidence_band.title())
    if result.warning_message:
        st.warning(result.warning_message)
    else:
        st.success("The model is highly confident in this prediction.")

    top_predictions = pd.DataFrame(
        [
            {
                "Rank": item.rank,
                "Digit": item.digit,
                "Probability": f"{item.probability:.2%}",
            }
            for item in result.top_predictions
        ]
    )
    st.write("Top-3 predictions")
    st.dataframe(top_predictions, use_container_width=True, hide_index=True)

    col_original, col_processed, col_saliency = st.columns(3)
    with col_original:
        st.caption("Original")
        st.image(result.original_image, use_container_width=True)
    with col_processed:
        st.caption("Processed 28x28")
        preview_result = type(
            "Preview",
            (),
            {"processed_image": result.processed_image},
        )
        st.image(create_preprocessing_preview(preview_result), use_container_width=True)
    with col_saliency:
        st.caption("Saliency")
        if result.saliency_map is not None:
            st.image(result.saliency_map, clamp=True, use_container_width=True)
            st.caption(SALIENCY_DISCLAIMER)


def render_history(history: list[HistoryRecord]) -> None:
    """Render prediction history.

    Args:
        history: Session prediction history.
    """
    if not history:
        return
    st.subheader("Session History")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Time": record.timestamp.strftime("%H:%M:%S"),
                    "Source": record.source_type,
                    "Digit": record.predicted_digit,
                    "Confidence": f"{record.confidence:.2%}",
                    "Band": record.confidence_band,
                    "Top 3": record.top_3_summary,
                }
                for record in history
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

