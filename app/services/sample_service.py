"""Sample digit helpers backed by MNIST test examples."""

from functools import lru_cache

import numpy as np
from PIL import Image

from app.services.image_service import prepare_uploaded_image
from app.services.prediction_service import _predict_from_preprocessed_result
from app.utils.exceptions import InvalidImageError
from src.data.data_loader import load_mnist_data


@lru_cache(maxsize=1)
def _load_test_samples() -> tuple[np.ndarray, np.ndarray]:
    """Load cached MNIST test samples."""
    (_, _), (x_test, y_test) = load_mnist_data()
    return x_test, y_test


def get_sample_digit(digit: int | None = None, index: int | None = None) -> tuple[Image.Image, int]:
    """Return a representative MNIST sample image.

    Args:
        digit: Optional target digit 0-9.
        index: Optional absolute MNIST test index.

    Returns:
        Tuple of PIL image and true digit label.

    Raises:
        InvalidImageError: If requested digit or index is invalid.
    """
    x_test, y_test = _load_test_samples()
    if index is not None:
        if index < 0 or index >= len(x_test):
            raise InvalidImageError("Sample index is outside the MNIST test set.")
        sample_index = index
    elif digit is not None:
        if digit < 0 or digit > 9:
            raise InvalidImageError("Sample digit must be between 0 and 9.")
        matches = np.where(y_test == digit)[0]
        sample_index = int(matches[0])
    else:
        sample_index = 0

    image = Image.fromarray(x_test[sample_index].astype("uint8"), mode="L")
    return image, int(y_test[sample_index])


def predict_sample_digit(digit: int | None = None, index: int | None = None):
    """Predict a dynamically loaded MNIST sample digit.

    Args:
        digit: Optional target digit.
        index: Optional test-set index.

    Returns:
        Prediction result.
    """
    image, _ = get_sample_digit(digit=digit, index=index)
    preprocessed = prepare_uploaded_image(image)
    preprocessed.source_type = "sample"
    return _predict_from_preprocessed_result(preprocessed)

