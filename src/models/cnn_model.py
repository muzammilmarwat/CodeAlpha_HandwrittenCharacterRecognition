"""CNN model definitions for handwritten digit recognition."""

from typing import Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Model

from src.utils.logging_config import setup_logger


logger = setup_logger(__name__)


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
) -> Model:
    """Build a baseline CNN architecture for MNIST classification.

    Args:
        input_shape: Shape of a single input image including channels.
        num_classes: Number of output classes.

    Returns:
        Untrained Keras model compiled for multiclass classification.
    """
    logger.info("Building baseline CNN with input shape %s", input_shape)
    model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="baseline_mnist_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
