"""Path helpers and artifact validation for the app layer."""

from pathlib import Path

from app.config import FINAL_REPORT_PATH, MODEL_CARD_PATH, MODEL_PATH
from app.utils.exceptions import ArtifactNotFoundError
from src.utils.paths import PROJECT_ROOT


def get_project_root() -> Path:
    """Return the repository root path."""
    return PROJECT_ROOT


def get_model_path() -> Path:
    """Return the trained model path."""
    return MODEL_PATH


def get_model_card_path() -> Path:
    """Return the model card path."""
    return MODEL_CARD_PATH


def get_final_report_path() -> Path:
    """Return the final model selection report path."""
    return FINAL_REPORT_PATH


def validate_required_artifacts() -> None:
    """Validate that required deployment artifacts exist.

    Raises:
        ArtifactNotFoundError: If one or more required artifacts are missing.
    """
    required_paths = [MODEL_PATH, MODEL_CARD_PATH, FINAL_REPORT_PATH]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = ", ".join(str(path) for path in missing_paths)
        raise ArtifactNotFoundError(f"Required artifact(s) missing: {formatted}")

