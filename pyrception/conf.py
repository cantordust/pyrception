import sys

# --------------------------------------
from decouple import config

# --------------------------------------
from loguru import logger

# --------------------------------------
import numpy as np

# --------------------------------------
import pyqtgraph as pg

pg.setConfigOptions(
    imageAxisOrder="row-major",
    useOpenGL=True,
    useNumba=True,
)

# --------------------------------------
from pyrception.visual.util.types import DType
from pyrception.visual.util.types import LogLevel

# Logger configuration
# ==================================================
# Enable colour tags in messages.
logger = logger.opt(ansi=True)

#: Configurable log level.
log_level = config("PYRCEPTION_LOG_LEVEL", default="INFO").upper()

if not LogLevel.contains(log_level):
    log_level = "INFO"

#: Colourise to the log output according to the level.
# TODO: Make the colours configurable.
for level in LogLevel:
    logger.level(level.name.upper(), color=level.value)

#: Log format.
log_format = config(
    "PYRCEPTION_LOG_FORMAT",
    default="Pyrception | <green>{time:YYYY-MM-DD@HH:mm:ss}</green> | <level>{message}</level>",
)

log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": log_format,
            "level": log_level,
        }
    ]
}

logger.configure(**log_config)
logger.info("Logger configured.")


# NumPy configuration
# ==================================================
#: Default dtype.
dtype = config("PYRCEPTION_NUMPY_DEFAULT_DTYPE", default="f32")
dtype = DType.get_value(dtype)
logger.info(f"Default NumPy data type: <yellow>{dtype}</yellow>.")
