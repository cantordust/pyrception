# --------------------------------------
from PySide6.QtCore import Slot

# --------------------------------------
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from pyqtgraph.parametertree import registerParameterType

# --------------------------------------
from pyrception.visual.utils.types import KernelParams
from pyrception.visual.utils.types import KernelFilter
from pyrception.gui.param import factory as pf


class KernelParameterGroup(GroupParameter):

    def __init__(self, **kwargs):

        kernel_params = kwargs.pop("kernel_params", {})
        super().__init__(**kwargs)

        # Active shape and filter for this RF
        # ==================================================
        # Create an instance
        kp = KernelParams(**kernel_params)

        # Shape
        # ==================================================
        self.shape = pf.make_enum("Shape", kp.shape)
        self.addChild(self.shape)

        # Kernel filter
        # ==================================================
        self.filter = pf.make_enum("Filter", kp.filter)
        self.filter.sigValueChanged.connect(self._make_filter_params)
        self.addChild(self.filter)

        # Kernel filter options
        # ==================================================
        self.prev_filter = self.filter.value()
        self.filters = None
        self.filter_factories = {
            KernelFilter.Uniform: self.make_uniform_filter_params,
            KernelFilter.Gaussian: self.make_gaussian_filter_params,
            KernelFilter.Gabor: self.make_gabor_filter_params,
        }
        self._make_filter_params(self.filter)

        # Scale
        # ==================================================
        self.scale = pf.make_float("Scale", kp.scale)
        self.addChild(self.scale)

        # Minimal size
        # ==================================================
        self.min_size = pf.make_sync_params(
            "Minimal size",
            pf.make_int,
            kp.min_size,
            [1, 16],
            name="min_size",
        )
        self.addChild(self.min_size)

        # Aspect
        # ==================================================
        self.aspect = pf.make_sync_params(
            "Aspect",
            pf.make_float,
            kp.aspect,
            [0.1, 10.0],
            step=0.05,
        )
        self.addChild(self.aspect)

    @Slot(Parameter)
    def _make_filter_params(
        self,
        param: Parameter,
    ):

        if self.filters is not None:
            self.removeChild(self.filters)

        index = self.childs.index(self.filter)
        self.filter_factories[param.value()]()
        self.insertChild(index + 1, self.filters)
        self.prev_filter = param.value()

    def make_uniform_filter_params(self):
        self.filters = pf.make_group("Uniform filter | Extra parameters", name="params")
        angle = pf.make_float(
            "Angle",
            0.0,
            limits=[0.0, 360.0],
            step=0.01,
        )
        self.filters.addChild(angle)

    def make_gaussian_filter_params(self):
        self.filters = pf.make_group(
            "Gaussian filter | Extra parameters", name="params"
        )
        angle = pf.make_float(
            "Angle",
            0.0,
            limits=[0.0, 360.0],
        )
        self.filters.addChild(angle)
        sd = pf.make_sync_params(
            "SD",
            cat=pf.make_float,
            value=0.37,
            limits=[0.01, 3.0],
            sync_kwargs={
                "step": 0.01,
            },
        )
        self.filters.addChild(sd)
        sd = pf.make_bool("Normalise")
        self.filters.addChild(sd)

    def make_gabor_filter_params(self):
        self.filters = pf.make_group("Gabor filter | Extra parameters", name="params")
        angle = pf.make_float("Angle")
        self.filters.addChild(angle)


registerParameterType("kernel_params", KernelParameterGroup)
