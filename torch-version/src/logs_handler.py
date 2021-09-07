import logging
from os import getenv

def get_log_level():
    """
    Return the log level according to LOGGING_LEVEL var env
    """

    return {
        "DEBUG" : logging.DEBUG,
        "INFO" : logging.INFO,
        "WARNING" : logging.WARNING,
        "ERROR" : logging.ERROR,
        "CRITICAL" : logging.CRITICAL,
    }.get(getenv("LOGGING_LEVEL"), logging.INFO)
