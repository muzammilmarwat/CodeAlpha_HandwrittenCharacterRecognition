"""Logging configuration for the Streamlit application."""

import logging


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_app_logging(level: int = logging.INFO) -> None:
    """Configure application logging once.

    Args:
        level: Logging level for application loggers.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    configure_app_logging()
    return logging.getLogger(name)

