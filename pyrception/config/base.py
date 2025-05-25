from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "pyrception"
LOCAL_DIR = ROOT_DIR / "local"
CONFIG_DIR = PACKAGE_DIR / "config"


class ConfBase(BaseSettings):
    """
    Configuration base class.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRCEPTION_",
        case_sensitive=True,
        validate_assignment=True,
    )
