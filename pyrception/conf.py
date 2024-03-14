import sys

# --------------------------------------
from decouple import config

# --------------------------------------
from loguru import logger

# --------------------------------------
import torch as pt

# Logger configuration
# ==================================================
log_format = config(
    "PYRCEPTION_LOG_FORMAT",
    default="Pyrception | <green>{time:DD-MMM-YYYY@HH:mm:ss}</green> | <level>{message}</level>",
)
log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": log_format,
        }
    ]
}

logger.configure(**log_config)
logger.info("Logger configured.")

# PyTorch configuration
# ==================================================
# Default tensor type
pt.set_default_dtype(pt.float16)

# Default device
device = config("PYRCEPTION_PYTORCH_DEVICE", default="cpu")

if device.split(":")[0] == "cuda":
    if pt.cuda.is_available():
        device = pt.device(device)
    else:
        logger.warning(f"CUDA is unavailable, using 'cpu' as a fallback.")
        device = pt.device("cpu")

else:
    device = pt.device("cpu")

logger.info(f"Default PyTorch device: '{device}'.")
