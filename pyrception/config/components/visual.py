import numpy as np
from pydantic import field_validator
from matplotlib import colors

from pyrception.config.base import ConfBase
from pyrception.utils.types import KernelShape
from pyrception.utils.types import KernelFilter
from pyrception.utils.types import RFArrangement

# from pyrception.visual.utils.types import KernelParams
from pyrception.config.validators import array_validator


class KernelConf(ConfBase):
    """
    Configuration for kernel parameters.

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

        extra (dict[str, tp.Any], optional):
            Additional kernel parameters.
            Defaults to {}.
    """

    shape: KernelShape = KernelShape.Elliptic
    filter: KernelFilter = KernelFilter.Uniform
    scale: float = 1.0
    min_size: np.ndarray = np.array([1, 1], dtype=np.int32)
    aspect: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    extra: dict = {}

    @field_validator("min_size", mode="before")
    @classmethod
    def _v_min_size(cls, value: str) -> np.ndarray:
        return array_validator(value)

    @field_validator("aspect", mode="before")
    @classmethod
    def _v_aspect(cls, value: str) -> np.ndarray:
        return array_validator(value, dtype=np.float32)


class RFConf(ConfBase):
    """
    Receptive field parameters.
    """

    name: str = "Receptive fields"
    sectors: int = 32
    extent: float = 1.0
    arrangement: RFArrangement = RFArrangement.LogPolar
    inverse: bool = False
    dense: bool = False
    feedback: bool = False
    kernel: KernelConf = KernelConf()


class PlotEntryConf(ConfBase):
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


class ImagePlotConf(PlotEntryConf):
    """
    Plotting parameters for images.
    """

    plottype: str = "image"
    cmap: str = "grey"
    norm: colors.Normalize = colors.Normalize()
    canvas: np.ndarray = None


class ScatterPlotConf(PlotEntryConf):
    """
    Plotting parameters for scatter plots.
    """

    plottype: str = "scatter"
    marker: str = "."
    colour: str = "#00ffff"
    size: float = 0.5
    # x: np.ndarray = None
    # y: np.ndarray = None


class Dim(ConfBase):
    """
    Dimension information.
    """

    height: int = 0
    width: int = 0
    depth: int = 1
    span: int = 0


class DimConf(ConfBase):
    """
    Dimension information for multiple views.
    """

    original: Dim = Dim()
    padded: Dim = Dim()
    padding: np.ndarray = np.array([0, 0, 0, 0])


class RFConf(ConfBase):
    name: str = "Receptive fields"
    sectors: int = 64
    arrangement: RFArrangement = RFArrangement.LogPolar
    shape: KernelShape = KernelShape.Elliptic
    filter: KernelFilter = KernelFilter.Uniform
    extent: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    scale: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    min_size: np.ndarray = np.array([1, 1], dtype=np.int32)
    angle: float = 0.0
    dense: bool = False
    feedback: bool = False
    phyllotactic: bool = False
    kernel_params: dict = {}

    @field_validator("min_size", mode="before")
    @classmethod
    def _v_min_size(cls, value: str) -> np.ndarray:
        return array_validator(value)

    @field_validator("extent", "scale", mode="before")
    @classmethod
    def _v_extent_scale(cls, value: str) -> np.ndarray:
        return array_validator(value, dtype=np.float32)


class LayerConf(ConfBase):
    rf: RFConf = RFConf()


class ReceptorLayerConf(LayerConf):
    pass


class HorizontalLayerConf(LayerConf):
    pass


class BipolarLayerConf(LayerConf):
    pass


class AmacrineLayerConf(LayerConf):
    pass


class GanglionLayerConf(LayerConf):
    pass


class VisualConf(ConfBase):
    kernel: KernelConf = KernelConf()
    receptor: ReceptorLayerConf = ReceptorLayerConf()
    horizontal: HorizontalLayerConf = HorizontalLayerConf()
    bipolar: BipolarLayerConf = BipolarLayerConf()
    amacrine: AmacrineLayerConf = AmacrineLayerConf()
    ganglion: GanglionLayerConf = GanglionLayerConf()
