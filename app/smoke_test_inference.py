"""Smoke test for Phase 4A inference services."""

import numpy as np

from app.services.prediction_service import predict_from_canvas
from app.utils.paths import validate_required_artifacts


def _synthetic_canvas_digit() -> np.ndarray:
    """Create a small white-on-black canvas-style digit for CI smoke tests."""
    canvas = np.zeros((280, 280, 4), dtype="uint8")
    canvas[..., 3] = 255
    canvas[55:225, 130:155, :3] = 255
    canvas[55:80, 90:170, :3] = 255
    return canvas


def main() -> None:
    """Run one end-to-end inference smoke test."""
    validate_required_artifacts()
    result = predict_from_canvas(_synthetic_canvas_digit())
    print("true_digit=synthetic_7")
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
