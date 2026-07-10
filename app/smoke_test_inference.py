"""Smoke test for Phase 4A inference services."""

from io import BytesIO

from app.services.prediction_service import predict_from_uploaded_image
from app.services.sample_service import get_sample_digit
from app.utils.paths import validate_required_artifacts


def main() -> None:
    """Run one end-to-end inference smoke test."""
    validate_required_artifacts()
    sample_image, true_digit = get_sample_digit(digit=7)
    buffer = BytesIO()
    sample_image.save(buffer, format="PNG")
    buffer.seek(0)

    result = predict_from_uploaded_image(buffer)
    print(f"true_digit={true_digit}")
    print(f"predicted_digit={result.predicted_digit}")
    print(f"confidence={result.confidence:.6f}")
    print(f"confidence_band={result.confidence_band}")
    print(
        "top_3="
        + ", ".join(
            f"{item.digit}:{item.probability:.6f}" for item in result.top_predictions
        )
    )
    print(f"saliency_generated={result.saliency_map is not None}")


if __name__ == "__main__":
    main()

