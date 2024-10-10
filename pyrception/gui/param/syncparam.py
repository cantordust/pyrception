# --------------------------------------
import typing as tp

# --------------------------------------
from PySide6.QtCore import Slot

# --------------------------------------
import numpy as np

# --------------------------------------
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from pyqtgraph.parametertree import registerParameterType

# --------------------------------------
from pyrception.visual.util.types import KernelParams
from pyrception.visual.util.types import KernelFilter
from pyrception.gui.param import factory as pf


class SyncParameter(GroupParameter):

    def __init__(
        self,
        cat: tp.Callable,
        value: tp.Iterable,
        limits: tp.Iterable,
        default: bool = True,
        names: tp.List = None,
        **kwargs
    ):

        sync_kwargs = kwargs.pop("sync_kwargs", {})
        super().__init__(**kwargs)

        if names is None:
            names = ['x', 'y']

        if not isinstance(value, tp.Iterable):
            value = [value, value]

        self.sync = pf.make_bool("Sync", value=default)
        self.addChild(self.sync)
        self.subgroup = []
        for name in names:
            p = cat(name, value=value[0], limits=limits, **sync_kwargs)
            self.subgroup.append(p)
            self.addChild(p)

        self.sync.sigValueChanged.connect(self._sync)
        self.sync.sigValueChanged.emit(self.sync, self.sync.value())

    @Slot(Parameter)
    def _sync(self, param: Parameter):

        subparam, subparams = self.subgroup[0], self.subgroup[1:]
        for sp in subparams:
            if param.value():
                sp.setReadonly()
                sp.setValue(subparam.value())
                subparam.sigValueChanged.connect(lambda: sp.setValue(subparam.value()))
            else:
                sp.setWritable()
                subparam.sigValueChanged.disconnect()

    def to_dict(self):

        return np.array([p.value() for p in self.subgroup])

registerParameterType("syncparam", SyncParameter)
