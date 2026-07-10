"""Session-ready prediction history helpers."""

from app.schemas.prediction_schema import HistoryRecord, PredictionResult


def create_history_record(prediction_result: PredictionResult) -> HistoryRecord:
    """Create a compact history record from a prediction result.

    Args:
        prediction_result: Full prediction result.

    Returns:
        History record.
    """
    top_3_summary = ", ".join(
        f"{item.digit}: {item.probability:.2%}" for item in prediction_result.top_predictions[:3]
    )
    return HistoryRecord(
        timestamp=prediction_result.timestamp,
        source_type=prediction_result.source_type,
        predicted_digit=prediction_result.predicted_digit,
        confidence=prediction_result.confidence,
        confidence_band=prediction_result.confidence_band,
        top_3_summary=top_3_summary,
    )


def append_history(
    history: list[HistoryRecord],
    record: HistoryRecord,
    max_items: int = 10,
) -> list[HistoryRecord]:
    """Append a record to in-memory prediction history.

    Args:
        history: Current history list.
        record: Record to append.
        max_items: Maximum number of records to keep.

    Returns:
        Updated history list.
    """
    return [record, *history][:max_items]


def clear_history() -> list[HistoryRecord]:
    """Return an empty prediction history list."""
    return []

