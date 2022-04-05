# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import enum


class View(enum.Enum):
    """
    Frame views for visual input.
    """

    Original = enum.auto()
    LocalMean = enum.auto()
    Normalised = enum.auto()
