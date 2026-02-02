import numpy as np
from matplotlib import colors
from dataclasses import dataclass
from dataclasses import field

# from pyrception.config.enums import KernelShape
# from pyrception.config.enums import KernelFilter
# from pyrception.config.enums import RFArrangement

# from pyrception.visual.utils.types import KernelParams
# from pyrception.config.validators import array_validator


@dataclass
class KernelConfig:
    """
    Configuration for kernel parameters.

    Attributes:

        shape: The shape of the kernel.
            NOTE: This is *not* the same as a tensor shape - rather,
            it is the geometric shape of the kernel (ellipse, rectangle, etc.).
        filter: The filter implemented by this kernel.
        scale: A scaling factor for kernels.
            Larger values result in kernels that may overlap more,
            while smaller values may result in kernels that leave gaps in
            the visual field.
        min_size: Minimal kernel size (usually restricted to the foveal region).
        aspect: Aspect ratio of the kernel.
        extra: Additional kernel parameters.
    """

    shape: np.ndarray
    scale: float = 1.0
    min_size: np.ndarray = field(default_factory=lambda: np.ones((2,), dtype=np.int32))
    aspect: np.ndarray = field(default_factory=lambda: np.ones((2,), dtype=np.float32))
    extra: dict = field(default_factory=lambda: dict())

@dataclass
class PlotEntryConfig:
    """
    Plotting parameters (base class).
    """

    data: np.ndarray
    plottype: str = None
    axis: bool = False
    spines: bool = False
    colourbar: bool = False
    clim: tuple[float, float] = (None, None)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    cmap: str = "grey"


@dataclass
class ImagePlotConfig(PlotEntryConfig):
    """
    Plotting parameters for images.
    """

    plottype: str = "image"
    cmap: str = "grey"
    norm: colors.Normalize = colors.Normalize()
    canvas: np.ndarray = None


@dataclass
class ScatterPlotConfig(PlotEntryConfig):
    """
    Plotting parameters for scatter plots.
    """

    plottype: str = "scatter"
    marker: str = "."
    colour: str = "#00ffff"
    size: float = 0.5


@dataclass
class Dim:
    """
    Dimension information.
    """

    height: int = 0
    width: int = 0
    depth: int = 1
    span: int = 0


@dataclass
class Dims:
    """
    Dimension information for multiple views.
    """

    original: Dim = field(default_factory=Dim)
    padded: Dim = field(default_factory=Dim)
    padding: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.uint32))
