"""The shared logger and its file/console handlers."""

import logging
import sys

from .config import LOGGER_NAME, LOG_FILE


def get_logger():
    """Return the single logger shared by every module in the package."""
    return logging.getLogger(LOGGER_NAME)


def setup_logging(run_dir):
    """Attach verbose file + console handlers to the shared logger."""
    logger = get_logger()
    log_path = run_dir / LOG_FILE
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info(f"Logging to {log_path}")
    return logger
