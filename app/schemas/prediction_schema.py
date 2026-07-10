"""Typed structures for image preprocessing and prediction results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
from PIL import Image


SourceType = Literal["upload", "canvas", "sample"]
ConfidenceBand = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class TopPrediction:
    """Single ranked class probability."""

    digit: int
    probability: float
    rank: int


@dataclass
class PreprocessedImageResult:
    """Image preprocessing output for model inference and UI preview."""

    model_tensor: np.ndarray
    processed_image: np.ndarray
    original_image: Image.Image
    source_type: SourceType
    warnings: list[str] = field(default_factory=list)
    digit_area_ratio: float = 0.0
    foreground_intensity: float = 0.0


@dataclass
class PredictionResult:
    """Complete prediction result for Streamlit display and history."""

    predicted_digit: int
    confidence: float
    confidence_band: ConfidenceBand
    top_predictions: list[TopPrediction]
    probabilities: list[float]
    original_image: Image.Image
    processed_image: np.ndarray
    saliency_map: np.ndarray | None
    warning_message: str | None
    model_name: str
    timestamp: datetime
    source_type: SourceType


@dataclass(frozen=True)
class HistoryRecord:
    """Session-ready prediction history record."""

    timestamp: datetime
    source_type: SourceType
    predicted_digit: int
    confidence: float
    confidence_band: ConfidenceBand
    top_3_summary: str

