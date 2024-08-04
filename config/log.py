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
    return logging.getLogger(name)


setup_logging()
