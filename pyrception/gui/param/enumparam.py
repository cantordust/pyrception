# --------------------------------------
import typing as tp

# --------------------------------------
from PySide6.QtCore import Slot

# --------------------------------------
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ListParameter
from pyqtgraph.parametertree import registerParameterType

# --------------------------------------
from pyrception.visual.util.types import AuxEnum


class EnumParameter(ListParameter):

    def __init__(
        self,
        value: AuxEnum,
        **kwargs,
    ):

        kwargs.update({"value": value})
        super().__init__(**kwargs)
        self.eclass = value.__class__
        self.sigValueChanged.connect(self._activate)

    @Slot(Parameter)
    def _activate(
        self,
        param: Parameter,
        value: tp.Union[AuxEnum, str],
    ):
        """
        Activate the right enumerator based on the parameter's value.

        Args:
            param (Parameter):
                Parameter used as a switch.

            value (str):
                Current value of the parameter.
        """

        value = self.eclass.get(value.name if isinstance(value, AuxEnum) else value)

        if value is not None:
            self.setValue(value)

registerParameterType("enum", EnumParameter)
