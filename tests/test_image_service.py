"""Tests for Phase 4A image preprocessing service."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.image_service import prepare_uploaded_image
from app.utils.exceptions import ImagePreprocessingError


def _digit_image(mode: str = "L", dark_on_light: bool = False) -> Image.Image:
    """Create a simple synthetic digit image."""
    background = 255 if dark_on_light else 0
    foreground = 0 if dark_on_light else 255
    image = Image.new("L", (80, 80), color=background)
    draw = ImageDraw.Draw(image)
    draw.line((25, 15, 55, 15, 35, 65), fill=foreground, width=9)
    return image.convert(mode)


def test_valid_grayscale_image_preprocessing() -> None:
    """Grayscale digit images should preprocess into model-ready tensors."""
    result = prepare_uploaded_image(_digit_image("L"))

    assert result.model_tensor.shape == (1, 28, 28, 1)
    assert result.processed_image.shape == (28, 28)
    assert result.model_tensor.min() >= 0
    assert result.model_tensor.max() <= 1


def test_rgb_image_preprocessing() -> None:
    """RGB images should preprocess into model-ready tensors."""
    result = prepare_uploaded_image(_digit_image("RGB"))

    assert result.model_tensor.shape == (1, 28, 28, 1)
    assert result.processed_image.max() > 0


def test_dark_on_light_polarity_correction() -> None:
    """Dark ink on light background should be converted to MNIST polarity."""
    result = prepare_uploaded_image(_digit_image("RGB", dark_on_light=True))

    assert result.model_tensor.shape == (1, 28, 28, 1)
    assert result.processed_image.max() > 0
    assert np.count_nonzero(result.processed_image) > 0


def test_blank_image_rejection() -> None:
    """Blank images should not be silently predicted."""
    blank = Image.new("L", (80, 80), color=255)

    with pytest.raises(ImagePreprocessingError):
        prepare_uploaded_image(blank)

