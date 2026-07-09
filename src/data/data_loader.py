"""MNIST data loading utilities."""

from typing import Any, Dict, Tuple

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


def get_dataset_summary(
    x_train: ImageArray,
    y_train: LabelArray,
    x_test: ImageArray,
    y_test: LabelArray,
) -> Dict[str, Any]:
    """Create a compact summary of MNIST shapes and class counts.

    Args:
        x_train: Training image array.
        y_train: Training label array.
        x_test: Test image array.
        y_test: Test label array.

    Returns:
        Dictionary containing dataset shapes and per-class counts.
    """
    class_names = get_class_names()
    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))

    return {
        "train_images_shape": list(x_train.shape),
        "train_labels_shape": list(y_train.shape),
        "test_images_shape": list(x_test.shape),
        "test_labels_shape": list(y_test.shape),
        "train_class_counts": {
            class_names[index]: int(count) for index, count in enumerate(train_counts)
        },
        "test_class_counts": {
            class_names[index]: int(count) for index, count in enumerate(test_counts)
        },
    }
