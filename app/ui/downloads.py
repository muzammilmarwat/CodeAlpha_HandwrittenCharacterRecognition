"""Download helpers for predictions and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.schemas.prediction_schema import HistoryRecord, PredictionResult
from app.utils.paths import get_final_report_path, get_model_card_path
from src.utils.paths import REPORTS_DIR


def prediction_summary_markdown(result: PredictionResult) -> str:
    """Create markdown prediction summary content."""
    top_3 = "\n".join(
        f"- Rank {item.rank}: digit {item.digit} ({item.probability:.4%})"
        for item in result.top_predictions
    )
    warning = result.warning_message or "No warning."
    return f"""# Prediction Summary

- Timestamp: {result.timestamp.isoformat(timespec="seconds")}
- Source type: {result.source_type}
- Predicted digit: {result.predicted_digit}
- Confidence: {result.confidence:.4%}
- Confidence band: {result.confidence_band}
- Model: {result.model_name}
- Warning: {warning}

## Top-3 Predictions
{top_3}

Educational and portfolio use only. Saliency indicates model sensitivity, not causal explanation.
"""


def prediction_summary_text(result: PredictionResult) -> str:
    """Create plain-text prediction summary content."""
    return prediction_summary_markdown(result).replace("# ", "").replace("## ", "")


def history_to_csv(history: list[HistoryRecord]) -> str:
    """Convert session history to CSV text."""
    frame = pd.DataFrame(
        [
            {
                "timestamp": record.timestamp.isoformat(timespec="seconds"),
                "source_type": record.source_type,
                "predicted_digit": record.predicted_digit,
                "confidence": record.confidence,
                "confidence_band": record.confidence_band,
                "top_3_summary": record.top_3_summary,
            }
            for record in history
        ]
    )
    return frame.to_csv(index=False)


def _read_file(path: Path) -> bytes:
    """Read file bytes for download."""
    return path.read_bytes()


def render_download_center(
    result: PredictionResult | None,
    history: list[HistoryRecord],
) -> None:
    """Render prediction and report downloads."""
    st.subheader("Download Center")
    if result is not None:
        st.download_button(
            "Download prediction summary (TXT)",
            prediction_summary_text(result),
            file_name="prediction_summary.txt",
        )
        st.download_button(
            "Download prediction summary (MD)",
            prediction_summary_markdown(result),
            file_name="prediction_summary.md",
        )
    else:
        st.caption("Run a prediction to enable prediction-summary downloads.")

    if history:
        st.download_button(
            "Download session history (CSV)",
            history_to_csv(history),
            file_name="session_prediction_history.csv",
            mime="text/csv",
        )

    report_files = [
        ("Model Card", get_model_card_path(), "model_card.md"),
        ("Final Model Selection Report", get_final_report_path(), "final_model_selection_report.md"),
        ("Evaluation Summary JSON", REPORTS_DIR / "evaluation_summary.json", "evaluation_summary.json"),
        ("Classification Report CSV", REPORTS_DIR / "classification_report.csv", "classification_report.csv"),
    ]
    for label, path, filename in report_files:
        if path.exists():
            st.download_button(label, _read_file(path), file_name=filename)
        else:
            st.caption(f"{label} is unavailable.")

