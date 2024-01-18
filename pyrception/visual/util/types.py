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

class KernelType(enum.Enum):
    """
    Receptive field organisation
    for building receptive fields.
    """

    Proportional = enum.auto()
    Gaussian = enum.auto()
    Gabor = enum.auto()
