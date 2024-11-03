# --------------------------------------
import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import enum


class AuxEnum(enum.Enum):

    @classmethod
    def get(
        cls: enum.Enum,
        key: str,
        default: enum.Enum = None,
    ):
        _key = key.lower()
        for dt in cls:
            if dt.name.lower() == _key:
                return dt
        return default

    @classmethod
    def get_value(
        cls: enum.Enum,
        key: str,
    ) -> tp.Optional[enum.Enum]:
        item = cls.get(key)
        return None if item is None else item.value

    @classmethod
    def contains(
        cls: enum.Enum,
        key: str,
    ) -> bool:
        """
        Check if a log key is valid.

        Args:
            key (str):
                The key to query for.

        Returns:
            bool:
                Indicator if the key was found.
        """
        return cls.get(key) is not None

    @classmethod
    def names(cls):
        return {o.name: o for o in cls}


class DType(AuxEnum):
    """
    NumPy data type.
    """

    F32 = np.float32
    F64 = np.double
    I8 = np.int8
    I16 = np.int16
    I32 = np.int32
    I64 = np.int64
    U8 = np.uint8


class LogLevel(AuxEnum):
    """
    Enum class that facilitates the configuration of logging levels.
    """

    Trace = "<light-blue>"
    Debug = "<cyan>"
    Info = "<light-green>"
    Success = "<green>"
    Warning = "<yellow>"
    Error = "<light-red>"
    Critical = "<red>"
