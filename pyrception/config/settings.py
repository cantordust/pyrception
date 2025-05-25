import os
import sys
from pathlib import Path

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import TomlConfigSettingsSource
from pydantic_settings import PydanticBaseSettingsSource

from typing import Type

from pyrception.config.base import CONFIG_DIR
from pyrception.config.base import ConfBase
from pyrception.config.components.log import LogConf
from pyrception.config.components.paths import PathConf
from pyrception.config.components.visual import VisualConf
from pyrception.config.components.numeric import NumConf


class Conf(ConfBase):
    """
    Main configuration.
    """

    paths: PathConf = PathConf()
    log: LogConf = LogConf()
    num: NumConf = NumConf()
    visual: VisualConf = VisualConf()

    # A class method that tries to identify all
    # the possible configuration files to load.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Default configuration file
        config_file = TomlConfigSettingsSource(
            settings_cls, CONFIG_DIR / "settings.toml"
        )

        try:
            local_settings = Path.cwd() / "settings.toml"
            if local_settings is not None and Path(local_settings).exists():
                # Check if there is a configuration file
                # for this specific class.
                config_file = TomlConfigSettingsSource(settings_cls, local_settings)
        except:
            pass

        return (config_file,)

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def expand_env_vars(cls, value: str) -> str:
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {
                k: (os.path.expandvars(v) if isinstance(v, str) else v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [(os.path.expandvars(v) if isinstance(v, str) else v) for v in value]

        return value


# Configuration object
# ==================================================
conf = Conf()

# Logger configuration
# ==================================================
# Enable colour tags in messages.
logger = logger.opt(colors=True)

#: Colourise to the log output according to the level.
for level, colour in conf.log.colour:
    logger.level(level.upper(), color=colour)


log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": conf.log.format,
            "level": conf.log.level,
        }
    ]
}

logger.configure(**log_config)
logger.info("Logger configured.")


# DType
# ==================================================
logger.info(f"NumPy | Data type: <yellow>{conf.num.numpy.dtype}</yellow>.")
