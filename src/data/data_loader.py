"""MNIST data loading utilities."""

from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.datasets import mnist

from src.utils.logging_config import setup_logger


logger = setup_logger(__name__)

ImageArray = np.ndarray
LabelArray = np.ndarray
MnistData = Tuple[Tuple[ImageArray, LabelArray], Tuple[ImageArray, LabelArray]]


def load_mnist_data() -> MnistData:
    """Load the MNIST handwritten digit dataset from Keras.

    Returns:
        Tuple containing training and test image-label pairs.
    """
    logger.info("Loading MNIST dataset from tensorflow.keras.datasets.mnist")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    logger.info(
        "Loaded MNIST dataset with %s training samples and %s test samples",
        x_train.shape[0],
        x_test.shape[0],
    )
    return (x_train, y_train), (x_test, y_test)


def get_class_names() -> Dict[int, str]:
    """Return human-readable class names for MNIST digits.

    Returns:
        Mapping of numeric class IDs to digit labels.
    """
    return {digit: str(digit) for digit in range(10)}
