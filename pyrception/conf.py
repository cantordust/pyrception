import sys

# --------------------------------------
from decouple import config

# --------------------------------------
from loguru import logger

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception.visual.util.types import DType
from pyrception.visual.util.types import LogLevel

# Logger configuration
# ==================================================
#: Configurable log level.
log_level = config(
    "PYRCEPTION_LOG_LEVEL", default="INFO"
).upper()

if LogLevel.contains(log_level) is None:
    log_level = "INFO"

# Add some colour to the log output.
LogLevel.colourise()

#: Configurable log format.
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


# PyTorch configuration
# ==================================================
#: Default dtype.
dtype = config("PYRCEPTION_PYTORCH_DEFAULT_DTYPE", default="float16")
pt.set_default_dtype(DType.get(dtype))
dtype = pt.get_default_dtype()
logger.info(f"Default PyTorch tensor type: '{dtype}'.")

#: Default PyTorch device.
device = config(
    "PYRCEPTION_PYTORCH_DEVICE",
    default="cpu",
)

# TODO: Abstract the following into a function
if device.split(":")[0] == "cuda":
    if pt.cuda.is_available():
        device = pt.device(device)
    else:
        logger.warning(f"CUDA is unavailable, using 'cpu' as a fallback.")
        device = pt.device("cpu")
else:
    device = pt.device("cpu")

logger.info(f"Default PyTorch device: '{device}'.")
