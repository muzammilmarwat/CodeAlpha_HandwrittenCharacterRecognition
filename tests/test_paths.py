"""Path and release configuration tests."""

from pathlib import Path

from app.config import APP_NAME, APP_VERSION, MODEL_PATH
from src.utils.paths import DATA_DIR, IMAGES_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR


def test_project_paths_are_pathlib_paths() -> None:
    """Configured paths should use pathlib objects."""
    paths = [PROJECT_ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, MODEL_PATH]

    assert all(isinstance(path, Path) for path in paths)


def test_release_paths_stay_inside_project_root() -> None:
    """Important release paths should resolve within the repository."""
    root = PROJECT_ROOT.resolve()
    paths = [DATA_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, MODEL_PATH]

    for path in paths:
        assert path.resolve().is_relative_to(root)


def test_app_release_metadata_is_final_candidate() -> None:
    """Release metadata should present the polished app identity."""
    assert APP_NAME == "Handwritten Digit Recognition Studio"
    assert "v1.0.0" in APP_VERSION
