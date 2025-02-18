# --------------------------------------
import enum

# --------------------------------------
import numpy as np

# --------------------------------------
from dataclasses import dataclass
from dataclasses import field


class InputType(enum.StrEnum):
    """
    Input type.
    """

    Image = enum.auto()
    Video = enum.auto()
    Events = enum.auto()


class RFArrangement(enum.StrEnum):
    """
    Receptive field distribution.
    """

    LogPolar = enum.auto()
    Cartesian = enum.auto()


class KernelFilter(enum.StrEnum):
    """
    Filter implemented by the receptive field.
    """

    Uniform = enum.auto()
    Gaussian = enum.auto()
    Gabor = enum.auto()


class KernelShape(enum.StrEnum):
    """
    Receptive field shape.
    """

    Elliptic = enum.auto()
    Rectangular = enum.auto()


@dataclass
class Dim:
    """
    A simple dataclass for holding dimension information.
    """

    height: int = 0
    width: int = 0
    depth: int = 1
    span: int = 0


@dataclass
class Dims:
    """
    A simple dataclass for holding dimension information
    for multiple views.
    """

    original: Dim = field(default_factory=Dim)
    padded: Dim = field(default_factory=Dim)
    padding: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0]))
