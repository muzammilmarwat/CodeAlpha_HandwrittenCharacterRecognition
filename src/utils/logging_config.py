"""Reusable logging configuration for project modules."""

import logging
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name. Uses the root logger when omitted.
        level: Logging level to apply.
        log_format: Format string for log messages.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
