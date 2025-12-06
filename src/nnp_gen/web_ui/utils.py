import logging
from contextlib import contextmanager

@contextmanager
def log_capture_to_file(logger_name: str, log_file: str):
    """
    Context manager to attach a FileHandler to a specific logger
    temporarily.
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
