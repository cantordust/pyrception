import sys
from loguru import logger
from pyrception.utils.enums import LogTint
from pyrception import conf

# Remove all existing handlers
logger.remove()

# Enable colour output
logger = logger.opt(colors=True)

# Colourise to the log output according to the level.
for tint in LogTint:
    logger.level(tint.name.upper(), color=tint.value)

logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": f"<m>Pyrception</m><level>{{extra[source]}}</level> | {{message}}",
            "level": conf.log.level,
        }
    ],
    extra={"source": ""},
)
