"""Release artifact integrity tests."""

from app.config import FINAL_REPORT_PATH, MODEL_CARD_PATH, MODEL_PATH


def test_required_release_artifacts_exist() -> None:
    """Required deployment artifacts should exist and be non-empty."""
    required_paths = [MODEL_PATH, MODEL_CARD_PATH, FINAL_REPORT_PATH]

    for path in required_paths:
        assert path.exists(), f"Missing required artifact: {path}"
        assert path.stat().st_size > 0, f"Empty required artifact: {path}"


def test_final_model_artifact_is_expected_keras_file() -> None:
    """The release should use the selected Keras model artifact."""
    assert MODEL_PATH.name == "mnist_cnn_baseline.keras"
    assert MODEL_PATH.suffix == ".keras"
    assert MODEL_PATH.stat().st_size > 1_000_000
