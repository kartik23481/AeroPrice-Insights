# src/utils/logger.py

import logging
import os

def get_logger(name: str, log_file: str):
    """
    Creates and returns a logger that writes to a specific file.
    """

    # ensure folder exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    
    # prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
