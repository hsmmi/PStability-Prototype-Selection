import logging
import logging.config
from config import LOGGING_CONFIG


def setup_logging():
    """Set up logging configuration."""
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception as e:
        print(f"Error configuring logging: {e}")
        raise


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name which is the module name."""
    logger = logging.getLogger(name)
    logger.info(f"LOGGER CREATED: {name}")
    return logger


setup_logging()

while True:
    # TEst
    logger = get_logger(__name__)
    logger.info("Test info message")
    logger.error("Test error message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    logger.critical("Test critical message")
