"""Model loading and validation service."""

from functools import lru_cache
from typing import Any

from tensorflow.keras.models import Model

from app.config import MODEL_NAME, MODEL_PATH
from app.utils.exceptions import ModelLoadingError
from app.utils.logging_config import get_logger
from src.inference.predict import load_trained_model


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model() -> Model:
    """Load the trained MNIST model with validation.

    Returns:
        Loaded Keras model.

    Raises:
        ModelLoadingError: If loading or signature validation fails.
    """
    try:
        if not MODEL_PATH.exists():
            raise ModelLoadingError(f"Model artifact not found: {MODEL_PATH}")
        model = load_trained_model()
        validate_model_signature(model)
        return model
    except ModelLoadingError:
        raise
    except Exception as exc:
        raise ModelLoadingError("Could not load the trained model artifact.") from exc


def validate_model_signature(model: Model) -> None:
    """Validate expected model input and output shapes.

    Args:
        model: Loaded Keras model.

    Raises:
        ModelLoadingError: If model signature is incompatible.
    """
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    if input_shape[-3:] != (28, 28, 1):
        raise ModelLoadingError(f"Unexpected model input shape: {input_shape}")
    if output_shape[-1] != 10:
        raise ModelLoadingError(f"Unexpected model output shape: {output_shape}")


def get_model_metadata() -> dict[str, Any]:
    """Return basic model metadata for UI display.

    Returns:
        Model metadata dictionary.
    """
    model = load_model()
    return {
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layer_count": len(model.layers),
    }

