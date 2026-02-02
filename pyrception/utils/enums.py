import enum


class StrEnum(enum.StrEnum):

    @classmethod
    def _missing_(cls, value: str | enum.StrEnum):
        if isinstance(value, cls):
            return value
        else:
            value = str(value)
        value = value.lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None


class LogTint(StrEnum):
    """
    Colours for different log levels.
    """

    trace = "<fg #ff7700>"
    debug = "<fg #0077ff>"
    info = "<fg #00ffff>"
    success = "<fg #00ff00>"
    warning = "<fg #ffff00>"
    error = "<fg #ff0000>"
    critical = "<fg #ff00ff>"

# Enums for the visual module
# ==================================================
class LogTint(StrEnum):
    trace: str = "<light-blue>"
    debug: str = "<cyan>"
    info: str = "<light-green>"
    success: str = "<green>"
    warning: str = "<yellow>"
    error: str = "<light-red>"
    critical: str = "<red>"


class InputType(StrEnum):
    """
    Input type.
    """

    Image = enum.auto()
    Video = enum.auto()
    Events = enum.auto()


class RFArrangement(StrEnum):
    """
    Receptive field distribution.
    """

    LogPolar = enum.auto()
    Cartesian = enum.auto()


class KernelFilter(StrEnum):
    """
    Filter implemented by the receptive field.
    """

    Uniform = enum.auto()
    Gaussian = enum.auto()
    Gabor = enum.auto()


class KernelShape(StrEnum):
    """
    Receptive field shape.
    """

    Elliptic = enum.auto()
    Rectangular = enum.auto()
