"""Tests for optional drawing-canvas service behavior."""

import numpy as np
import pytest

from app.services.image_service import prepare_canvas_image
from app.services.prediction_service import predict_from_canvas
from app.utils.exceptions import ImagePreprocessingError


def _valid_canvas_digit() -> np.ndarray:
    """Create a simple white-on-black RGBA canvas digit."""
    canvas = np.zeros((280, 280, 4), dtype="uint8")
    canvas[..., 3] = 255
    canvas[55:225, 130:155, :3] = 255
    canvas[55:80, 90:170, :3] = 255
    return canvas


def test_blank_canvas_rejection() -> None:
    """Blank black canvas input should be rejected."""
    canvas = np.zeros((280, 280, 4), dtype="uint8")
    canvas[..., 3] = 255

    with pytest.raises(ImagePreprocessingError):
        prepare_canvas_image(canvas)


def test_valid_canvas_preprocessing_shape_and_range() -> None:
    """Valid canvas drawings should preprocess into normalized model tensors."""
    result = prepare_canvas_image(_valid_canvas_digit())

    assert result.model_tensor.shape == (1, 28, 28, 1)
    assert result.model_tensor.min() >= 0
    assert result.model_tensor.max() <= 1
    assert result.processed_image.shape == (28, 28)
    assert result.source_type == "canvas"


def test_canvas_prediction_contract() -> None:
    """Canvas predictions should return the standard prediction contract."""
    result = predict_from_canvas(_valid_canvas_digit())

    assert 0 <= result.predicted_digit <= 9
    assert len(result.top_predictions) == 3
    assert len(result.probabilities) == 10
    assert result.confidence_band in {"low", "medium", "high"}
    assert result.saliency_map is not None
    assert result.saliency_map.shape == (28, 28)
