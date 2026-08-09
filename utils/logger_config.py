import logging
from utils.definitions import LOGS_DIR, PIPELINE_LOG_PATH


def setup_logging():
    # Create logs directory if it does not exist
    LOGS_DIR.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        style="%",
    )

    # File handler
    file_handler = logging.FileHandler(
        PIPELINE_LOG_PATH,
        mode="a",
        encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)




