"""Tests for Phase 4B UI-independent helpers."""

from datetime import datetime, timedelta

import numpy as np

from app.schemas.prediction_schema import HistoryRecord, PredictionResult, TopPrediction
from app.services.history_service import append_history, clear_history, create_history_record
from app.services.prediction_service import get_top_k_predictions, interpret_confidence
from app.services.sample_service import predict_sample_digit
from app.services.saliency_service import generate_saliency_map
from app.ui.downloads import history_to_csv, prediction_summary_markdown
from app.services.model_service import load_model


def _dummy_prediction() -> PredictionResult:
    """Create a lightweight prediction result for helper tests."""
    from PIL import Image

    processed = np.zeros((28, 28), dtype="uint8")
    return PredictionResult(
        predicted_digit=7,
        confidence=0.95,
        confidence_band="high",
        top_predictions=[
            TopPrediction(digit=7, probability=0.95, rank=1),
            TopPrediction(digit=3, probability=0.03, rank=2),
            TopPrediction(digit=9, probability=0.02, rank=3),
        ],
        probabilities=[0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.95, 0.0, 0.02],
        original_image=Image.fromarray(processed, mode="L").convert("RGBA"),
        processed_image=processed,
        saliency_map=processed.astype("float32"),
        warning_message=None,
        model_name="mnist_cnn_baseline",
        timestamp=datetime.now(),
        source_type="sample",
    )


def test_confidence_band_boundaries() -> None:
    """Confidence thresholds should match product rules."""
    assert interpret_confidence(0.69) == "low"
    assert interpret_confidence(0.70) == "medium"
    assert interpret_confidence(0.89) == "medium"
    assert interpret_confidence(0.90) == "high"


def test_top_k_predictions_are_sorted() -> None:
    """Top-k helper should return sorted ranked classes."""
    top = get_top_k_predictions(np.array([0.1, 0.7, 0.2]), k=2)

    assert [item.digit for item in top] == [1, 2]
    assert [item.rank for item in top] == [1, 2]


def test_history_append_limit_and_clear() -> None:
    """History helpers should enforce max item count and clear cleanly."""
    result = _dummy_prediction()
    records = []
    for index in range(12):
        record = create_history_record(result)
        object.__setattr__(record, "timestamp", datetime.now() + timedelta(seconds=index))
        records = append_history(records, record, max_items=10)

    assert len(records) == 10
    assert clear_history() == []


def test_prediction_summary_generation() -> None:
    """Prediction summary downloads should contain key prediction fields."""
    summary = prediction_summary_markdown(_dummy_prediction())

    assert "Predicted digit: 7" in summary
    assert "Confidence: 95.0000%" in summary
    assert "Top-3 Predictions" in summary


def test_history_csv_generation() -> None:
    """History CSV downloads should include session columns."""
    record = HistoryRecord(
        timestamp=datetime.now(),
        source_type="sample",
        predicted_digit=7,
        confidence=0.95,
        confidence_band="high",
        top_3_summary="7: 95.00%",
    )
    csv_text = history_to_csv([record])

    assert "predicted_digit" in csv_text
    assert "top_3_summary" in csv_text


def test_sample_prediction_works() -> None:
    """Sample prediction should return a valid digit result."""
    result = predict_sample_digit(digit=7)

    assert 0 <= result.predicted_digit <= 9
    assert len(result.top_predictions) == 3
    assert result.source_type == "sample"


def test_saliency_shape_for_sample_prediction() -> None:
    """Saliency generation should return a 28x28 map."""
    result = predict_sample_digit(digit=7)
    model = load_model()
    tensor = result.processed_image.astype("float32").reshape(1, 28, 28, 1) / 255.0
    saliency = generate_saliency_map(model, tensor, result.predicted_digit)

    assert saliency.shape == (28, 28)

