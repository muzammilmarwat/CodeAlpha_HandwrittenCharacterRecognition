"""Tests for prediction service outputs."""

from io import BytesIO

import numpy as np

from app.schemas.prediction_schema import PredictionResult
from app.services.prediction_service import predict_from_uploaded_image
from app.services.sample_service import get_sample_digit


def _sample_upload_buffer(digit: int = 7) -> BytesIO:
    """Create an in-memory upload buffer from an MNIST sample."""
    image, _ = get_sample_digit(digit=digit)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_prediction_result_contains_top_3_probabilities() -> None:
    """Prediction output should include stable top-3 probabilities."""
    result = predict_from_uploaded_image(_sample_upload_buffer())

    assert isinstance(result, PredictionResult)
    assert len(result.top_predictions) == 3
    assert len(result.probabilities) == 10
    assert result.top_predictions[0].probability >= result.top_predictions[1].probability
    assert result.top_predictions[1].probability >= result.top_predictions[2].probability


def test_prediction_probabilities_and_confidence_are_valid() -> None:
    """Prediction probabilities should be normalized and labels valid."""
    result = predict_from_uploaded_image(_sample_upload_buffer())

    assert np.isclose(sum(result.probabilities), 1.0, atol=1e-4)
    assert 0 <= result.predicted_digit <= 9
    assert result.confidence_band in {"low", "medium", "high"}
    assert result.saliency_map is not None
    assert result.saliency_map.shape == (28, 28)

