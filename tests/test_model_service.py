"""Tests for model and artifact validation services."""

from app.services.model_service import get_model_metadata, load_model
from app.utils.paths import validate_required_artifacts


def test_artifact_validation_passes() -> None:
    """Required model/report artifacts should exist."""
    validate_required_artifacts()


def test_model_loads_and_signature_is_valid() -> None:
    """Saved model should load with the expected MNIST signature."""
    model = load_model()
    metadata = get_model_metadata()

    assert model.input_shape[-3:] == (28, 28, 1)
    assert model.output_shape[-1] == 10
    assert metadata["model_name"] == "mnist_cnn_baseline"

