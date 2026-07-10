"""Prediction orchestration service for app inputs."""

from datetime import datetime
from typing import BinaryIO

import numpy as np

from app.config import (
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    TOP_K,
)
from app.schemas.prediction_schema import (
    ConfidenceBand,
    PredictionResult,
    PreprocessedImageResult,
    SourceType,
    TopPrediction,
)
from app.services.image_service import (
    load_uploaded_image,
    prepare_canvas_image,
    prepare_uploaded_image,
)
from app.services.model_service import load_model
from app.services.saliency_service import generate_saliency_map
from app.utils.exceptions import PredictionError


def interpret_confidence(confidence: float) -> ConfidenceBand:
    """Map numeric confidence to a confidence band.

    Args:
        confidence: Top prediction probability.

    Returns:
        Confidence band label.
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def confidence_message(confidence_band: ConfidenceBand) -> str:
    """Return user-facing confidence guidance.

    Args:
        confidence_band: Confidence band label.

    Returns:
        Guidance message.
    """
    if confidence_band == "high":
        return "The model is highly confident in this prediction."
    if confidence_band == "medium":
        return "The prediction is plausible, but review the image quality and alternatives."
    return "The model is uncertain. Try centering the digit, increasing contrast, or drawing more clearly."


def get_top_k_predictions(probabilities: np.ndarray, k: int = TOP_K) -> list[TopPrediction]:
    """Return top-k class probabilities sorted descending.

    Args:
        probabilities: Probability vector for digits 0-9.
        k: Number of ranked predictions to return.

    Returns:
        Ranked top predictions.
    """
    vector = np.asarray(probabilities).reshape(-1)
    top_indices = np.argsort(vector)[::-1][:k]
    return [
        TopPrediction(digit=int(index), probability=float(vector[index]), rank=rank)
        for rank, index in enumerate(top_indices, start=1)
    ]


def _build_warning_message(
    confidence_band: ConfidenceBand,
    top_predictions: list[TopPrediction],
    preprocessing_warnings: list[str],
) -> str | None:
    """Build a combined confidence and preprocessing warning."""
    messages = []
    if confidence_band != "high":
        messages.append(confidence_message(confidence_band))
    if len(top_predictions) >= 2 and (
        top_predictions[0].probability - top_predictions[1].probability < 0.10
    ):
        messages.append("Top predictions are close; treat this result as ambiguous.")
    messages.extend(preprocessing_warnings)
    return " ".join(messages) if messages else None


def _predict_from_preprocessed_result(
    preprocessed: PreprocessedImageResult,
) -> PredictionResult:
    """Predict from a preprocessed image result."""
    try:
        model = load_model()
        probabilities = model.predict(preprocessed.model_tensor, verbose=0)[0].astype("float64")
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        confidence_band = interpret_confidence(confidence)
        top_predictions = get_top_k_predictions(probabilities, k=TOP_K)
        saliency_map = generate_saliency_map(model, preprocessed.model_tensor, predicted_digit)
        warning_message = _build_warning_message(
            confidence_band,
            top_predictions,
            preprocessed.warnings,
        )

        return PredictionResult(
            predicted_digit=predicted_digit,
            confidence=confidence,
            confidence_band=confidence_band,
            top_predictions=top_predictions,
            probabilities=[float(value) for value in probabilities.tolist()],
            original_image=preprocessed.original_image,
            processed_image=preprocessed.processed_image,
            saliency_map=saliency_map,
            warning_message=warning_message,
            model_name=MODEL_NAME,
            timestamp=datetime.now(),
            source_type=preprocessed.source_type,
        )
    except Exception as exc:
        if exc.__class__.__name__.endswith("Error"):
            raise
        raise PredictionError("Prediction failed. Please try a clearer digit image.") from exc


def predict_from_uploaded_image(uploaded_file: BinaryIO) -> PredictionResult:
    """Run prediction from an uploaded image file.

    Args:
        uploaded_file: Streamlit uploaded file or file-like object.

    Returns:
        Prediction result.
    """
    image = load_uploaded_image(uploaded_file)
    preprocessed = prepare_uploaded_image(image)
    return _predict_from_preprocessed_result(preprocessed)


def predict_from_canvas(canvas_array: np.ndarray) -> PredictionResult:
    """Run prediction from a drawing canvas array.

    Args:
        canvas_array: Canvas image array.

    Returns:
        Prediction result.
    """
    preprocessed = prepare_canvas_image(canvas_array)
    return _predict_from_preprocessed_result(preprocessed)


def predict_from_preprocessed_tensor(
    tensor: np.ndarray,
    source_type: SourceType,
) -> PredictionResult:
    """Predict directly from a model-ready tensor.

    This is mainly useful for tests or advanced Phase 4B UI flows.

    Args:
        tensor: Model input tensor shaped as (1, 28, 28, 1).
        source_type: Source label.

    Returns:
        Prediction result.
    """
    processed_image = np.clip(tensor.reshape(28, 28) * 255.0, 0, 255).astype("uint8")
    from PIL import Image

    preprocessed = PreprocessedImageResult(
        model_tensor=tensor.astype("float32"),
        processed_image=processed_image,
        original_image=Image.fromarray(processed_image, mode="L").convert("RGBA"),
        source_type=source_type,
    )
    return _predict_from_preprocessed_result(preprocessed)

