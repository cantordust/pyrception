from __future__ import annotations

# --------------------------------------
import typing as tp

# --------------------------------------
from loguru import logger

# --------------------------------------
from dataclasses import dataclass
from dataclasses import field

# --------------------------------------
from matplotlib import colors

# --------------------------------------
import numpy as np

# --------------------------------------
import enum

# --------------------------------------
from PySide6.QtCore import Slot

# --------------------------------------
from pyqtgraph.parametertree import Parameter

"""
===================================[ NOTE ]===================================
Enumerators
==============================================================================
"""


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


class InputType(AuxEnum):
    """
    Input type.
    """

    Image = enum.auto()
    Video = enum.auto()
    Events = enum.auto()


class RFArrangement(AuxEnum):
    """
    Receptive field distribution.
    """

    LogPolar = enum.auto()
    Cartesian = enum.auto()


class KernelFilter(AuxEnum):
    """
    Filter implemented by the receptive field.
    """

    Uniform = enum.auto()
    Gaussian = enum.auto()
    Gabor = enum.auto()


class KernelShape(AuxEnum):
    """
    Receptive field shape.
    """

    Elliptic = enum.auto()
    Rectangular = enum.auto()


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


"""
===================================[ NOTE ]===================================
Dataclasses
==============================================================================
"""


@dataclass
class KernelParams:
    """
    A dataclass for kernel parameters.

    Attributes:

        shape (KernelShape, optional):
            The shape of the kernel. Defaults to KernelShape.Elliptic.
            NOTE: This is *not* the same as a tensor shape - rather,
            it is the geometric (2D) shape of the kernel.

        filter (KernelFilter, optional):
            The filter response type for receptive fields in this layer.
            Defaults to KernelFilter.Uniform.

        scale (float, optional):
            A scaling factor for kernels.
            Defaults to 1.0.
            Larger values result in kernels that may overlap more,
            while smaller values may result in kernels that leave gaps in
            the visual field.

        min_size (np.ndarray, optional):
            Minimal kernel size (usually restricted to the foveal region).
            Defaults to (1, 1).

        aspect (np.ndarray, optional):
            Aspect ratio of the kernel.
            Defaults to (1.0, 1.0).

        params (tp.Dict[str, tp.Any], optional):
            Additional kernel parameters.
            Defaults to {}.
    """

    shape: KernelShape = KernelShape.Elliptic
    filter: KernelFilter = KernelFilter.Uniform
    scale: float = 1.0
    min_size: np.ndarray = field(
        default_factory=lambda: np.array([1, 1], dtype=np.int32)
    )
    aspect: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=np.float32)
    )
    params: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def __post_init__(self):

        # Validate the min_size attribute
        if isinstance(self.min_size, (float, int)):
            self.min_size = [self.min_size, self.min_size]
        if isinstance(self.min_size, np.ndarray):
            self.min_size = self.min_size.astype(np.int32)
        else:
            self.min_size = np.array(self.min_size, dtype=np.int32)

        # Validate the aspect attribute
        if isinstance(self.aspect, (float, int)):
            self.aspect = [self.aspect, self.aspect]
        if isinstance(self.aspect, np.ndarray):
            self.aspect = self.aspect.astype(np.float32)
        else:
            self.aspect = np.array(self.aspect, dtype=np.float32)


@dataclass
class RFParams:
    '''
    A dataclass for receptive field parameters
    '''

    substrate: np.ndarray = None
    sectors: int = 32
    extent: float = 1.0
    arrangement: RFArrangement = RFArrangement.LogPolar
    inverse: bool = False
    dense: bool = False
    create_feedback: bool = False
    name: str = "Receptive fields"
    kernel_params: KernelParams = None


@dataclass
class PlotEntry:
    """
    A convenience class for plotting parameters.
    """

    data: np.ndarray
    plottype: str = None
    axis: bool = False
    spines: bool = False
    colourbar: bool = False
    clim: tp.Tuple[float, float] = (None, None)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""


@dataclass
class ImagePlot(PlotEntry):
    """
    A convenience class for plotting parameters for images.
    """

    plottype: str = "image"
    cmap: str = "grey"
    norm: colors.Normalize = colors.Normalize()
    canvas: np.ndarray = None


@dataclass
class ScatterPlot(PlotEntry):
    """
    A convenience class for plotting parameters for scatter plots.
    """

    plottype: str = "scatter"
    marker: str = "."
    colour: str = "#00ffff"
    size: float = 0.5
    x: np.ndarray = None
    y: np.ndarray = None


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
