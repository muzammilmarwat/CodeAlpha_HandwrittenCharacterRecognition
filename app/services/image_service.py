"""Robust image preprocessing for uploaded, canvas, and sample digit inputs."""

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

from app.config import INPUT_IMAGE_SIZE, MAX_UPLOAD_SIZE_MB
from app.schemas.prediction_schema import PreprocessedImageResult, SourceType
from app.utils.exceptions import ImagePreprocessingError, InvalidImageError
from app.utils.logging_config import get_logger


logger = get_logger(__name__)


def load_uploaded_image(uploaded_file: BinaryIO) -> Image.Image:
    """Load an uploaded image file as a PIL image.

    Args:
        uploaded_file: Streamlit uploaded file or file-like object.

    Returns:
        Loaded PIL image.

    Raises:
        InvalidImageError: If the file is too large or cannot be decoded.
    """
    try:
        if hasattr(uploaded_file, "size") and uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise InvalidImageError(f"Upload must be {MAX_UPLOAD_SIZE_MB} MB or smaller.")
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        image = Image.open(BytesIO(data))
        image.load()
        return image
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError("Could not read the image. Please upload a PNG, JPG, or JPEG file.") from exc


def validate_image(image: Image.Image) -> None:
    """Validate that an image has usable dimensions.

    Args:
        image: PIL image to validate.

    Raises:
        InvalidImageError: If image dimensions are invalid.
    """
    if image.width <= 0 or image.height <= 0:
        raise InvalidImageError("Image dimensions are invalid.")
    if image.width < 5 or image.height < 5:
        raise InvalidImageError("Image is too small to contain a readable digit.")


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert RGB, RGBA, grayscale, or palette images to grayscale.

    Transparent images are composited over white before grayscale conversion.

    Args:
        image: Input PIL image.

    Returns:
        Grayscale PIL image.
    """
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return ImageOps.grayscale(image)


def detect_and_correct_polarity(image_array: np.ndarray) -> np.ndarray:
    """Convert images to MNIST polarity: light digit on dark background.

    Args:
        image_array: Grayscale image array in [0, 255].

    Returns:
        Polarity-corrected image array.
    """
    border_pixels = np.concatenate(
        [
            image_array[0, :],
            image_array[-1, :],
            image_array[:, 0],
            image_array[:, -1],
        ]
    )
    if float(np.median(border_pixels)) > 127:
        return 255 - image_array
    return image_array


def was_polarity_adjusted(image_array: np.ndarray) -> bool:
    """Return whether an image likely needs inversion to match MNIST polarity.

    Args:
        image_array: Grayscale image array in [0, 255].

    Returns:
        True when the border background appears light.
    """
    border_pixels = np.concatenate(
        [
            image_array[0, :],
            image_array[-1, :],
            image_array[:, 0],
            image_array[:, -1],
        ]
    )
    return float(np.median(border_pixels)) > 127


def remove_uniform_background(image_array: np.ndarray) -> np.ndarray:
    """Normalize contrast after polarity correction.

    Args:
        image_array: Grayscale image array.

    Returns:
        Contrast-normalized uint8 array.
    """
    array = image_array.astype("float32")
    array = array - float(np.percentile(array, 5))
    array = np.clip(array, 0, None)
    max_value = float(array.max())
    if max_value > 0:
        array = array / max_value * 255.0
    return array.astype("uint8")


def threshold_image(image_array: np.ndarray) -> np.ndarray:
    """Create a clean foreground mask while preserving digit intensity.

    Args:
        image_array: Contrast-normalized image array.

    Returns:
        Thresholded image array with background set to zero.
    """
    if image_array.max() < 15:
        raise ImagePreprocessingError("The image appears blank or too faint to classify.")
    threshold = max(20.0, float(np.mean(image_array) + 0.5 * np.std(image_array)))
    cleaned = np.where(image_array >= threshold, image_array, 0)
    if np.count_nonzero(cleaned) < 8:
        raise ImagePreprocessingError("No clear digit foreground was detected.")
    return cleaned.astype("uint8")


def extract_digit_bounding_box(image_array: np.ndarray) -> tuple[int, int, int, int]:
    """Extract the bounding box of non-background digit pixels.

    Args:
        image_array: Thresholded image array.

    Returns:
        Bounding box as left, upper, right, lower.

    Raises:
        ImagePreprocessingError: If no digit foreground exists.
    """
    ys, xs = np.where(image_array > 0)
    if xs.size == 0 or ys.size == 0:
        raise ImagePreprocessingError("No digit foreground was detected.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_digit(image_array: np.ndarray) -> np.ndarray:
    """Crop an image to the detected digit foreground.

    Args:
        image_array: Thresholded image array.

    Returns:
        Cropped digit image array.
    """
    left, upper, right, lower = extract_digit_bounding_box(image_array)
    return image_array[upper:lower, left:right]


def resize_preserving_aspect_ratio(image_array: np.ndarray, target_box: int = 20) -> np.ndarray:
    """Resize a digit to fit inside a square target box.

    Args:
        image_array: Cropped digit image array.
        target_box: Maximum resized width or height.

    Returns:
        Resized digit array.
    """
    height, width = image_array.shape
    if height == 0 or width == 0:
        raise ImagePreprocessingError("Digit crop is empty.")
    scale = target_box / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = Image.fromarray(image_array, mode="L")
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype="uint8")


def center_digit_on_canvas(image_array: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    """Center a resized digit on a square canvas using center of mass.

    Args:
        image_array: Resized digit image array.
        canvas_size: Output canvas size.

    Returns:
        Centered 28x28 image array.
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype="uint8")
    height, width = image_array.shape
    top = max(0, (canvas_size - height) // 2)
    left = max(0, (canvas_size - width) // 2)
    canvas[top : top + height, left : left + width] = image_array

    ys, xs = np.where(canvas > 0)
    if xs.size == 0:
        raise ImagePreprocessingError("Centered canvas has no digit foreground.")

    center_y = int(round(float(np.average(ys, weights=canvas[ys, xs]))))
    center_x = int(round(float(np.average(xs, weights=canvas[ys, xs]))))
    shift_y = canvas_size // 2 - center_y
    shift_x = canvas_size // 2 - center_x
    shifted = np.zeros_like(canvas)

    src_y_start = max(0, -shift_y)
    src_y_end = min(canvas_size, canvas_size - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(canvas_size, canvas_size - shift_x)
    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = canvas[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
    ]
    return shifted


def normalize_for_model(image_array: np.ndarray) -> np.ndarray:
    """Normalize a 28x28 image and reshape for model input.

    Args:
        image_array: Processed 28x28 image array.

    Returns:
        Model tensor shaped as (1, 28, 28, 1).
    """
    tensor = image_array.astype("float32") / 255.0
    return tensor.reshape(1, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 1)


def _prepare_image(image: Image.Image, source_type: SourceType) -> PreprocessedImageResult:
    """Run the full app preprocessing pipeline."""
    try:
        validate_image(image)
        original_image = image.convert("RGBA")
        grayscale = convert_to_grayscale(image)
        gray_array = np.asarray(grayscale, dtype="uint8")
        polarity_adjusted = was_polarity_adjusted(gray_array)
        corrected = detect_and_correct_polarity(gray_array)
        normalized = remove_uniform_background(corrected)
        thresholded = threshold_image(normalized)
        foreground_count = int(np.count_nonzero(thresholded))
        digit_area_ratio = foreground_count / float(thresholded.size)
        cropped = crop_to_digit(thresholded)
        resized = resize_preserving_aspect_ratio(cropped)
        centered = center_digit_on_canvas(resized)
        model_tensor = normalize_for_model(centered)

        warnings: list[str] = []
        if digit_area_ratio < 0.01:
            warnings.append("Digit occupies very little of the image; prediction may be unstable.")
        if float(centered.max()) < 80:
            warnings.append("Digit foreground is faint; try increasing contrast.")

        return PreprocessedImageResult(
            model_tensor=model_tensor,
            processed_image=centered,
            original_image=original_image,
            source_type=source_type,
            warnings=warnings,
            digit_area_ratio=digit_area_ratio,
            foreground_intensity=float(centered.max()) / 255.0,
            original_dimensions=image.size,
            processed_dimensions=centered.shape[::-1],
            tensor_shape=tuple(model_tensor.shape),
            polarity_adjusted=polarity_adjusted,
        )
    except (InvalidImageError, ImagePreprocessingError):
        raise
    except Exception as exc:
        raise ImagePreprocessingError("Image preprocessing failed. Try a clearer centered digit.") from exc


def prepare_uploaded_image(image: Image.Image) -> PreprocessedImageResult:
    """Prepare an uploaded image for model prediction.

    Args:
        image: PIL image loaded from upload.

    Returns:
        Preprocessed image result.
    """
    return _prepare_image(image, source_type="upload")


def prepare_canvas_image(canvas_array: np.ndarray) -> PreprocessedImageResult:
    """Prepare a canvas image array for model prediction.

    Args:
        canvas_array: Canvas RGBA/RGB/grayscale array.

    Returns:
        Preprocessed image result.
    """
    if canvas_array is None or np.asarray(canvas_array).size == 0:
        raise InvalidImageError("Canvas is empty.")
    image = Image.fromarray(np.asarray(canvas_array).astype("uint8"))
    return _prepare_image(image, source_type="canvas")


def create_preprocessing_preview(result: PreprocessedImageResult) -> Image.Image:
    """Create a displayable preview image from preprocessing output.

    Args:
        result: Preprocessed image result.

    Returns:
        PIL image preview.
    """
    return Image.fromarray(result.processed_image, mode="L").resize((140, 140), Image.Resampling.NEAREST)
