# --------------------------------------
from pyrception.config.base import ConfBase

# --------------------------------------
from pydantic import field_validator


class LogColour(ConfBase):
    trace: str = "<light-blue>"
    debug: str = "<cyan>"
    info: str = "<light-green>"
    success: str = "<green>"
    warning: str = "<yellow>"
    error: str = "<light-red>"
    critical: str = "<red>"


class LogConf(ConfBase):
    """
    Logger configuration.
    """

    verbose: bool = True
    format: str = (
        "<magenta>Pyrception</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    )
    level: str = "INFO"
    colour: LogColour = LogColour()

    @field_validator("level", mode="before", check_fields=False)
    @classmethod
    def _v_level(cls, value: str) -> str:
        return value.upper()