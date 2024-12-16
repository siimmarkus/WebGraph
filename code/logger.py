"""Logging utility"""

import logging

LOGGER_NAME = "webgraph"
LOGGER_FORMAT = "%(levelname)s %(message)s"
LOGGER_LEVEL = logging.DEBUG


def configure_logger() -> logging.Logger:
    """Configure the logger used by Webgraph."""
    formatter = logging.Formatter(LOGGER_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOGGER_LEVEL)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = configure_logger()
