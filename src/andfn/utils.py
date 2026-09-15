import logging.config

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    # "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "minimal": {"format": "%(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "minimal",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

logger = logging.getLogger(__name__)

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def configure_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.debug("Logging is configured.")


def set_log_level(level: str, all_handlers: bool = False) -> None:
    """Set the logging level for the andfn logger.

    Parameters
    ----------
    level : str
        The logging level to set. Options are 'DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL'.
    all_handlers : bool
        If True, set the log level for all handlers. Default is False for which only the
        StreamHandler is set.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise TypeError(f"Invalid log level: {level}")
    if all_handlers:
        for handler in logging.getLogger().handlers:
            handler.setLevel(numeric_level)
    else:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(numeric_level)
    logger.info(f"Log level set to {level}")


def enable_file_logging(
    name: str = "andfn.log", filemode: str = "w", loglevel: str = "DEBUG"
) -> None:
    """Enable logging to a file named 'andfn.log' in the current directory.

    Parameters
    ----------
    name : str
        The name of the log file. Default is 'andfn.log'.
    filemode : str
        The mode to open the log file. Default is 'w' (write mode).
        Other common mode is 'a' (append mode).
    loglevel : str
        The logging level for the file handler. Default is 'DEBUG'. Options are
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """

    root_logger = logging.getLogger()

    # check if the name has an extension, if not add .log
    if not name.endswith(".log"):
        name += ".log"

    # Avoid adding duplicate handlers
    import os

    abs_name = os.path.abspath(name)

    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == abs_name
        ):
            logger.warning(f"File logging already enabled for {name}")
            return

    root_logger = logging.getLogger()

    # Check if the log level is valid
    loglevel = loglevel.upper()
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise TypeError(f"Invalid log level: {numeric_level}")
    # Add file handler to root logger
    file_handler = logging.FileHandler(name, mode=filemode, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root_logger.addHandler(file_handler)

    logger.info(f"File logging enabled: {name} ({loglevel})")
