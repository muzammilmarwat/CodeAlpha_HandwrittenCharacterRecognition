"""Saliency generation service for inference-time explanations."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from app.utils.exceptions import SaliencyGenerationError


SALIENCY_DISCLAIMER = (
    "Saliency highlights input pixels that most influenced model sensitivity. "
    "It is not a causal explanation."
)


def generate_saliency_map(model: Model, input_tensor: np.ndarray, predicted_class: int) -> np.ndarray:
    """Generate a normalized saliency map for a model input.

    Args:
        model: Trained Keras model.
        input_tensor: Model input shaped as (1, 28, 28, 1).
        predicted_class: Class index used as the gradient target.

    Returns:
        Normalized 2D saliency array.

    Raises:
        SaliencyGenerationError: If saliency generation fails.
    """
    try:
        tensor = tf.convert_to_tensor(input_tensor.astype("float32"))
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            predictions = model(tensor, training=False)
            class_score = predictions[:, predicted_class]
        gradients = tape.gradient(class_score, tensor)
        if gradients is None:
            return np.zeros((28, 28), dtype="float32")
        saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0].numpy()
        min_value = float(np.min(saliency))
        max_value = float(np.max(saliency))
        if np.isclose(max_value, min_value):
            return np.zeros_like(saliency, dtype="float32")
        return ((saliency - min_value) / (max_value - min_value)).astype("float32")
    except Exception as exc:
        raise SaliencyGenerationError("Could not generate saliency map for this prediction.") from exc


def overlay_saliency_on_processed_image(processed_image: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
    """Create a simple saliency overlay array.

    Args:
        processed_image: Processed 28x28 image array.
        saliency_map: Normalized 28x28 saliency map.

    Returns:
        RGB overlay array in [0, 1].
    """
    base = processed_image.astype("float32") / 255.0
    saliency = np.clip(saliency_map.astype("float32"), 0.0, 1.0)
    overlay = np.zeros((28, 28, 3), dtype="float32")
    overlay[..., 0] = np.maximum(base, saliency)
    overlay[..., 1] = base * 0.65
    overlay[..., 2] = base * 0.65
    return np.clip(overlay, 0.0, 1.0)

