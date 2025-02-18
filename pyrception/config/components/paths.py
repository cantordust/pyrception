# --------------------------------------
from pathlib import Path

# --------------------------------------
from pyrception.config.base import ConfBase
from pyrception.config.base import ROOT_DIR
from pyrception.config.base import PACKAGE_DIR
from pyrception.config.base import CONFIG_DIR
from pyrception.config.base import LOCAL_DIR

class PathConf(ConfBase):
    """
    Core path configuration.
    """

    root_dir: Path = ROOT_DIR
    package_dir: Path = PACKAGE_DIR
    config_dir: Path = CONFIG_DIR
    local_dir: Path = LOCAL_DIR
