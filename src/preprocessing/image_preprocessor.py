"""Image preprocessing helpers for MNIST and future uploads."""

from typing import Tuple

import numpy as np

from src.utils.logging_config import setup_logger


logger = setup_logger(__name__)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale image pixel values to the range [0, 1].

    Args:
        images: Input image array with pixel values typically in [0, 255].

    Returns:
        Float32 image array normalized to [0, 1].
    """
    logger.debug("Normalizing images with shape %s", images.shape)
    return images.astype("float32") / 255.0


def reshape_for_cnn(images: np.ndarray, image_shape: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """Reshape grayscale image batches for CNN input.

    Args:
        images: Image batch shaped as (n_samples, height, width).
        image_shape: Expected two-dimensional image shape.

    Returns:
        Image batch shaped as (n_samples, height, width, 1).
    """
    logger.debug("Reshaping images from %s for CNN input", images.shape)
    return images.reshape((-1, image_shape[0], image_shape[1], 1))


def preprocess_mnist_images(images: np.ndarray) -> np.ndarray:
    """Apply standard preprocessing for MNIST images.

    Args:
        images: Raw MNIST image batch.

    Returns:
        Normalized image batch with a channel dimension.
    """
    normalized_images = normalize_images(images)
    return reshape_for_cnn(normalized_images)


def preprocess_uploaded_image_placeholder(image: object) -> None:
    """Placeholder for future uploaded-image preprocessing.

    Args:
        image: Uploaded image object from a future application interface.

    Returns:
        None. This function will be implemented in the deployment phase.
    """
    logger.info("Uploaded image preprocessing is not implemented yet.")
    return None
