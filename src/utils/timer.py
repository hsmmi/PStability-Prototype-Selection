import time
from config.log import get_logger
from contextlib import contextmanager
import logging

logger = get_logger("mylogger")


@contextmanager
def measure_time(label="Execution time", level="INFO"):
    """
    Measure the execution time of a block of code.

    Parameters:
    label (str): Label for the execution time.
    level (str): Logging level.
    """
    level = getattr(logging, level)
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    logger.log(level, f"{label}: {execution_time:.6f} seconds")
