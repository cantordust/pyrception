from typing import *

# --------------------------------------
from loguru import logger

# --------------------------------------
from dataclasses import dataclass
from dataclasses import field

# --------------------------------------
import numpy as np

# --------------------------------------
import enum

# --------------------------------------
from pprint import pp

# --------------------------------------
import torch as pt


class RFArrangement(enum.Enum):
    """
    Receptive field distribution.
    """

    LogPolar = enum.auto()
    Cartesian = enum.auto()


class KernelFilter(enum.Enum):
    """
    Receptive field organisation.
    """

    Flat = enum.auto()
    Gaussian = enum.auto()
    Gabor = enum.auto()


class KernelShape(enum.Enum):
    """
    Receptive field shape.
    """

    Elliptic = enum.auto()
    Rectangular = enum.auto()


class DType(enum.Enum):
    """
    Pytorch tensor dtype.

    Raises:
        ValueError:
            Error raised if the provided dtype is invalid.

    Returns:
        pt.dtype:
            PyTorch dtype.
    """

    F16 = pt.half
    F32 = pt.float
    F64 = pt.double
    I8 = pt.int8
    I16 = pt.int16
    I32 = pt.int32
    I64 = pt.int64
    U8 = pt.uint8

    @staticmethod
    def get(dtype: str):
        _dtype = dtype.lower()
        for dt in DType:
            if dt.name.lower() == _dtype:
                return dt.value

        raise ValueError(f"Invalid dtype '{dtype}'")


class LogLevel(enum.Enum):
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

    @staticmethod
    def contains(level: str) -> Optional[str]:
        """
        Check if a log level is valid.

        Returns:
            Optional[str]:
                The level (as an uppercase string)
                or None if the level is invalid.
        """
        level = level.upper()
        for lvl in LogLevel:
            if lvl.name.upper() == level:
                return level

        return

    @staticmethod
    def colourise():
        """
        Colourise the log output.
        """

        for level in LogLevel:
            logger.level(level.name.upper(), color=level.value)


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
            Defaults to KernelFilter.Flat.

        scale (float), optional):
            A scaling factor for kernels.
            Defaults to 1.0.
            Larger values result in kernels that may overlap more,
            while smaller values may result in kernels that leave gaps in
            the visual field.

        min_size (pt.Tensor, optional):
            Minimal kernel size (usually restricted to the foveal region).
            Defaults to (1, 1).

        aspect (pt.Tensor, optional):
            Aspect ratio of the kernel.
            Defaults to (1.0, 1.0).

        params (Dict[str, Any], optional):
            Additional kernel parameters.
            Defaults to {}.
    """

    shape: KernelShape = KernelShape.Elliptic
    filter: KernelFilter = KernelFilter.Flat
    scale: float = 1.0
    min_size: pt.Tensor = field(
        default_factory=lambda: pt.tensor([1, 1], dtype=pt.int32)
    )
    aspect: pt.Tensor = field(
        default_factory=lambda: pt.tensor([1.0, 1.0], dtype=pt.float32)
    )
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):

        # Validate the min_size attribute
        if isinstance(self.min_size, (float, int)):
            self.min_size = [self.min_size, self.min_size]
        if not isinstance(self.min_size, pt.Tensor):
            self.min_size = pt.tensor(self.min_size, dtype=pt.int32)

        # Validate the aspect attribute
        if isinstance(self.aspect, (float, int)):
            self.aspect = [self.aspect, self.aspect]
        if not isinstance(self.aspect, pt.Tensor):
            self.aspect = pt.tensor(self.aspect, dtype=pt.float32)

@dataclass
class PlotEntry:
    '''
    A convenience class for plotting parameters.
    '''

    data: np.ndarray
    plottype: str = None
    axis: bool = False
    spines: bool = False
    colourbar: bool = False
    clim: Tuple[float, float] = (None, None)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""

@dataclass
class ImagePlot(PlotEntry):
    '''
    A convenience class for plotting parameters for images.
    '''

    plottype: str = "image"

@dataclass
class ScatterPlot(PlotEntry):
    '''
    A convenience class for plotting parameters for scatter plots.
    '''

    plottype: str = "scatter"
    marker: str = "."
    colour: str = "#00ffff"
    size: float = 0.5

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
