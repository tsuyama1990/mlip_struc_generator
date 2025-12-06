import logging
import threading
from contextlib import contextmanager

class ThreadLogFilter(logging.Filter):
    """
    Filter that only allows log records from a specific thread.
    """
    def __init__(self, thread_id):
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record):
        return record.thread == self.thread_id

@contextmanager
def log_capture_to_file(logger_name: str, log_file: str):
    """
    Context manager to attach a FileHandler to a specific logger
    temporarily, filtered to the current thread.
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add filter to ensure we only capture logs from this thread
    # This prevents global log hijacking when multiple threads are active
    handler.addFilter(ThreadLogFilter(threading.get_ident()))

    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
