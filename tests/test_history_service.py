"""Tests for prediction history service behavior."""

from datetime import datetime

import numpy as np
from PIL import Image

from app.schemas.prediction_schema import PredictionResult, TopPrediction
from app.services.history_service import append_history, create_history_record


def _result(digit: int, confidence: float) -> PredictionResult:
    """Create a compact prediction result for history tests."""
    processed = np.zeros((28, 28), dtype="uint8")
    return PredictionResult(
        predicted_digit=digit,
        confidence=confidence,
        confidence_band="high",
        top_predictions=[
            TopPrediction(digit=digit, probability=confidence, rank=1),
            TopPrediction(digit=(digit + 1) % 10, probability=0.05, rank=2),
            TopPrediction(digit=(digit + 2) % 10, probability=0.02, rank=3),
        ],
        probabilities=[0.0] * 10,
        original_image=Image.fromarray(processed, mode="L").convert("RGBA"),
        processed_image=processed,
        saliency_map=None,
        warning_message=None,
        model_name="mnist_cnn_baseline",
        timestamp=datetime(2026, 7, 10, 12, 0, digit),
        source_type="sample",
    )


def test_create_history_record_preserves_prediction_summary() -> None:
    """History records should keep compact prediction metadata."""
    record = create_history_record(_result(digit=8, confidence=0.9876))

    assert record.predicted_digit == 8
    assert record.confidence == 0.9876
    assert record.source_type == "sample"
    assert "8: 98.76%" in record.top_3_summary


def test_append_history_keeps_newest_first() -> None:
    """New history records should appear before older records."""
    first = create_history_record(_result(digit=1, confidence=0.9))
    second = create_history_record(_result(digit=2, confidence=0.8))

    history = append_history([], first)
    history = append_history(history, second)

    assert [record.predicted_digit for record in history] == [2, 1]
