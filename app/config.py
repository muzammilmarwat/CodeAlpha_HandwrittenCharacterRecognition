"""Application configuration for Phase 4A inference."""

from pathlib import Path

from src.utils.paths import MODELS_DIR, REPORTS_DIR


APP_NAME = "CodeAlpha Handwritten Digit Recognition"
APP_VERSION = "0.4.0-alpha"
MODEL_NAME = "mnist_cnn_baseline"
MODEL_PATH = MODELS_DIR / "mnist_cnn_baseline.keras"
SUPPORTED_IMAGE_TYPES = ("png", "jpg", "jpeg")
INPUT_IMAGE_SIZE = (28, 28)
LOW_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.90
TOP_K = 3
MAX_UPLOAD_SIZE_MB = 5
DEFAULT_CANVAS_SIZE = 280
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"
FINAL_REPORT_PATH = REPORTS_DIR / "final_model_selection" / "final_model_selection_report.md"

