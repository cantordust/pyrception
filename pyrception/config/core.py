from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from environs import env


@dataclass
class LogConf:
    level = env.str("PYRCEPTION_LOG_LEVEL", "INFO").upper()

@dataclass
class Conf:
    log: LogConf = field(default_factory=LogConf)


conf = Conf()
