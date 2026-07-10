"""Tests for downloadable report and summary helpers."""

from datetime import datetime

import numpy as np
from PIL import Image

from app.schemas.prediction_schema import PredictionResult, TopPrediction
from app.ui.downloads import history_to_csv, prediction_summary_markdown, prediction_summary_text
from app.utils.paths import get_final_report_path, get_model_card_path


def _prediction_result() -> PredictionResult:
    """Create a deterministic prediction result for download tests."""
    processed = np.zeros((28, 28), dtype="uint8")
    return PredictionResult(
        predicted_digit=4,
        confidence=0.9123,
        confidence_band="high",
        top_predictions=[
            TopPrediction(digit=4, probability=0.9123, rank=1),
            TopPrediction(digit=9, probability=0.0522, rank=2),
            TopPrediction(digit=7, probability=0.0211, rank=3),
        ],
        probabilities=[0.0, 0.0, 0.0, 0.0, 0.9123, 0.0, 0.0, 0.0211, 0.0, 0.0522],
        original_image=Image.fromarray(processed, mode="L").convert("RGBA"),
        processed_image=processed,
        saliency_map=processed.astype("float32"),
        warning_message=None,
        model_name="mnist_cnn_baseline",
        timestamp=datetime(2026, 7, 10, 12, 0, 0),
        source_type="canvas",
    )


def test_prediction_download_text_contains_release_fields() -> None:
    """Prediction summaries should expose the fields users need to save."""
    result = _prediction_result()

    markdown = prediction_summary_markdown(result)
    text = prediction_summary_text(result)

    assert "Predicted digit: 4" in markdown
    assert "Confidence: 91.2300%" in markdown
    assert "Top-3 Predictions" in markdown
    assert "Prediction Summary" in text
    assert "#" not in text


def test_empty_history_csv_still_has_headers() -> None:
    """Empty history downloads should remain valid CSV text."""
    csv_text = history_to_csv([])

    assert csv_text.strip() == ""


def test_release_report_download_paths_exist() -> None:
    """Report download paths should point to real release artifacts."""
    assert get_model_card_path().exists()
    assert get_final_report_path().exists()
