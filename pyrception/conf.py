import sys

# --------------------------------------
from decouple import config

# --------------------------------------
from loguru import logger

LOG_FORMAT = config(
    "PYRCEPTION_LOG_FORMAT",
    # default="==[ <green>{time:DD-MMM-YYYY@HH:mm:ss}</green> ] <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    # default="==[ <green>{time:DD-MMM-YYYY@HH:mm:ss}</green> ] <level>{level:<8}</level> | <level>{message}</level>",
    default="==[ <green>{time:DD-MMM-YYYY@HH:mm:ss}</green> | <level>{message}</level>",
)
LOG_CONFIG = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": LOG_FORMAT,
        }
    ]
}

logger.configure(**LOG_CONFIG)
