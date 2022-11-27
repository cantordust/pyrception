import enum


class View(enum.Enum):
    """
    Frame views for visual input.
    """

    Original = enum.auto()
    ReceptorMean = enum.auto()
    ReceptorPadded = enum.auto()
    ReceptorAdapted = enum.auto()
    BipolarMean = enum.auto()
    BipolarOn = enum.auto()
    BipolarOff = enum.auto()
    BipolarCombined = enum.auto()
    GanglionOnOff = enum.auto()
    GanglionOffOn = enum.auto()
    Composite = enum.auto()
    OnOffEvents = enum.auto()


class RFSizeDist(enum.Enum):
    """
    Kernel size distribution type
    for building receptive fields.
    """

    Flat = enum.auto()
    Gaussian = enum.auto()
    LogPolar = enum.auto()


class RFType(enum.Enum):
    """
    Receptive field organisation
    for building receptive fields.
    """

    CentreSurround = enum.auto()
    Proportional = enum.auto()
    Gabor = enum.auto()
    Flat = enum.auto()
