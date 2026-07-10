"""Custom exceptions for the Streamlit inference application."""


class AppError(Exception):
    """Base exception for safe Streamlit display messages."""


class ArtifactNotFoundError(AppError):
    """Raised when a required model or report artifact is missing."""


class InvalidImageError(AppError):
    """Raised when an uploaded or drawn image is invalid."""


class ImagePreprocessingError(AppError):
    """Raised when image preprocessing cannot create a valid model input."""


class ModelLoadingError(AppError):
    """Raised when the trained model cannot be loaded or validated."""


class PredictionError(AppError):
    """Raised when prediction cannot be completed."""


class SaliencyGenerationError(AppError):
    """Raised when saliency generation fails."""


class CanvasNotAvailableError(AppError):
    """Raised when drawing-canvas functionality is unavailable."""

