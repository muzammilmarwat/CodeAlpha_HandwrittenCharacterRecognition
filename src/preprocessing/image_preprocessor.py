"""Image preprocessing helpers for MNIST and future uploads."""

from typing import Tuple

import numpy as np
from tensorflow.keras.utils import to_categorical

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


def reshape_for_cnn(
    images: np.ndarray,
    image_shape: Tuple[int, int] = (28, 28),
) -> np.ndarray:
    """Reshape grayscale image batches for CNN input.

    Args:
        images: Image batch shaped as (n_samples, height, width).
        image_shape: Expected two-dimensional image shape.

    Returns:
        Image batch shaped as (n_samples, height, width, 1).
    """
    logger.debug("Reshaping images from %s for CNN input", images.shape)
    return images.reshape((-1, image_shape[0], image_shape[1], 1))


def one_hot_encode_labels(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """One-hot encode integer class labels.

    Args:
        labels: Integer label array.
        num_classes: Number of output classes.

    Returns:
        One-hot encoded label array.
    """
    logger.debug("One-hot encoding labels with shape %s", labels.shape)
    return to_categorical(labels, num_classes=num_classes)


def preprocess_mnist_images(images: np.ndarray) -> np.ndarray:
    """Apply standard preprocessing for MNIST image arrays.

    Args:
        images: Raw MNIST image batch or a single image.

    Returns:
        Normalized image array with a channel dimension.
    """
    normalized_images = normalize_images(images)
    return reshape_for_cnn(normalized_images)


def preprocess_mnist_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess MNIST train and test arrays for CNN training.

    Args:
        x_train: Raw training images.
        y_train: Raw training labels.
        x_test: Raw test images.
        y_test: Raw test labels.
        num_classes: Number of output classes for one-hot encoding.

    Returns:
        Tuple of processed training images, training labels, test images, and
        test labels.
    """
    logger.info("Preprocessing MNIST images and labels")
    x_train_processed = preprocess_mnist_images(x_train)
    x_test_processed = preprocess_mnist_images(x_test)
    y_train_processed = one_hot_encode_labels(y_train, num_classes=num_classes)
    y_test_processed = one_hot_encode_labels(y_test, num_classes=num_classes)
    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


def preprocess_uploaded_image_placeholder(image: object) -> None:
    """Placeholder for future uploaded-image preprocessing.

    Args:
        image: Uploaded image object from a future application interface.

    Returns:
        None. This function will be implemented in the deployment phase.
    """
    logger.info("Uploaded image preprocessing is not implemented yet.")
    return None
