# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import enum


class View(enum.Enum):
    """
    Frame views for visual input.
    """

    Debug = enum.auto()
    Original = enum.auto()
    ReceptorMean = enum.auto()
    ReceptorPadded = enum.auto()
    ReceptorAdapted = enum.auto()
    BipolarMean = enum.auto()
    BipolarOn = enum.auto()
    BipolarOff = enum.auto()
    GanglionOnOff = enum.auto()
    GanglionOffOn = enum.auto()
    OpticalFlow = enum.auto()


class KernelSizeDist(enum.Enum):
    """
    Kernel size distribution type
    for building receptive fields.
    """

    Flat = enum.auto()
    Gaussian = enum.auto()
    LogPolar = enum.auto()


class KernelWeightDist(enum.Enum):
    """
    Kernel weight distribution type
    for building receptive fields.
    """

    Flat = enum.auto()
    Gaussian = enum.auto()
    Proportional = enum.auto()
