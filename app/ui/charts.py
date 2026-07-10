"""Visualization helpers for Streamlit UI pages."""

import pandas as pd
import streamlit as st

from app.schemas.prediction_schema import PredictionResult


def top_3_probability_chart(result: PredictionResult) -> None:
    """Render top-3 probability horizontal bars."""
    data = pd.DataFrame(
        {
            "Digit": [str(item.digit) for item in result.top_predictions],
            "Probability": [item.probability for item in result.top_predictions],
        }
    ).set_index("Digit")
    st.bar_chart(data, horizontal=True, height=190)


def all_class_probability_chart(result: PredictionResult) -> None:
    """Render all digit probabilities in stable digit order."""
    data = pd.DataFrame(
        {
            "Digit": [str(index) for index in range(10)],
            "Probability": result.probabilities,
        }
    ).set_index("Digit")
    st.bar_chart(data, height=260)


def confidence_visual(result: PredictionResult) -> None:
    """Render confidence progress and top-1/top-2 margin."""
    top_1 = result.top_predictions[0].probability
    top_2 = result.top_predictions[1].probability if len(result.top_predictions) > 1 else 0.0
    margin = top_1 - top_2
    st.progress(result.confidence, text=f"Confidence: {result.confidence:.2%}")
    st.progress(max(0.0, min(1.0, margin)), text=f"Top-1 vs Top-2 margin: {margin:.2%}")
    if margin < 0.10:
        st.warning("Top predictions are close. Review alternatives before trusting the result.")


def per_class_performance_chart(per_class_df: pd.DataFrame) -> None:
    """Render per-class accuracy chart."""
    data = per_class_df[["digit", "accuracy"]].copy()
    data["digit"] = data["digit"].astype(str)
    st.bar_chart(data.set_index("digit"), height=260)


def confusion_pair_chart(confusion_pairs_df: pd.DataFrame) -> None:
    """Render common confusion pairs chart."""
    if confusion_pairs_df.empty:
        st.info("No confusion pairs available.")
        return
    data = confusion_pairs_df.head(10).copy()
    data["pair"] = data.apply(
        lambda row: f"{int(row['true_digit'])}->{int(row['predicted_digit'])}",
        axis=1,
    )
    st.bar_chart(data[["pair", "count"]].set_index("pair"), horizontal=True, height=300)

