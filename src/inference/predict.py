"""Inference helpers for the trained MNIST CNN model."""

from functools import lru_cache
from typing import Any, Dict

import numpy as np
from tensorflow.keras.models import Model, load_model

from src.preprocessing.image_preprocessor import normalize_images, reshape_for_cnn
from src.utils.logging_config import setup_logger
from src.utils.paths import MODELS_DIR


logger = setup_logger(__name__)

MODEL_PATH = MODELS_DIR / "mnist_cnn_baseline.keras"


@lru_cache(maxsize=1)
def load_trained_model() -> Model:
    """Load the saved MNIST CNN model.

    Returns:
        Loaded Keras model.

    Raises:
        FileNotFoundError: If the trained model artifact does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run training first."
        )

    logger.info("Loading trained model from %s", MODEL_PATH)
    return load_model(MODEL_PATH)


def _prepare_images_for_prediction(images: np.ndarray) -> np.ndarray:
    """Prepare one or more MNIST-like images for model prediction.

    Args:
        images: Image array shaped as (28, 28), (28, 28, 1), (n, 28, 28),
            or (n, 28, 28, 1).

    Returns:
        Normalized image batch shaped as (n, 28, 28, 1).
    """
    image_array = np.asarray(images)

    if image_array.ndim == 2:
        image_array = image_array.reshape(1, 28, 28)
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = image_array.reshape(1, 28, 28, 1)

    if image_array.ndim == 3:
        image_array = reshape_for_cnn(image_array)
    elif image_array.ndim != 4:
        raise ValueError(
            "Expected image shape (28, 28), (28, 28, 1), "
            "(n, 28, 28), or (n, 28, 28, 1)."
        )

    if image_array.shape[1:] != (28, 28, 1):
        raise ValueError(f"Expected MNIST image shape (28, 28, 1), got {image_array.shape[1:]}.")

    if image_array.max() > 1.0:
        image_array = normalize_images(image_array)
    else:
        image_array = image_array.astype("float32")

    return image_array


def predict_batch(images: np.ndarray) -> Dict[str, Any]:
    """Predict digits for a batch of MNIST-like images.

    Args:
        images: Batch of images to classify.

    Returns:
        Dictionary with predicted digits, confidences, and probabilities.
    """
    prepared_images = _prepare_images_for_prediction(images)
    model = load_trained_model()
    probabilities = model.predict(prepared_images, verbose=0)
    predicted_digits = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    return {
        "predicted_digits": predicted_digits.astype(int).tolist(),
        "confidences": confidences.astype(float).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
    }


def predict_digit(image_array: np.ndarray) -> Dict[str, Any]:
    """Predict the digit represented by a single MNIST-like image.

    Args:
        image_array: Single image shaped as (28, 28) or (28, 28, 1).

    Returns:
        Dictionary with the predicted digit, confidence, and probabilities.
    """
    batch_result = predict_batch(image_array)
    return {
        "predicted_digit": batch_result["predicted_digits"][0],
        "confidence": batch_result["confidences"][0],
        "probabilities": batch_result["probabilities"][0],
    }
