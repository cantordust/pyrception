# --------------------------------------
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from pyqtgraph.parametertree import registerParameterType

# --------------------------------------
from pyrception.visual.utils.types import RFParams
from pyrception.visual.utils.types import KernelParams
from pyrception.gui.param.kernel import KernelParameterGroup
from pyrception.gui.param import factory as pf


class RFParameterGroup(GroupParameter):

    def __init__(self, **kwargs):

        rf_params = kwargs.pop("rf_params", {})
        kernel_params = rf_params.pop("kernel_params", {})
        super().__init__(**kwargs)

        # Active shape and filter for this RF
        # ==================================================
        # Create an instance
        rp = RFParams(**rf_params)

        # Extent
        # ==================================================
        self.extent = pf.make_float("Extent", rp.extent)
        self.addChild(self.extent)

        # Arrangment
        # ==================================================
        self.arrangement = pf.make_enum("Arrangement", rp.arrangement)
        self.addChild(self.arrangement)

        # Inverse toggle
        # ==================================================
        self.inverse = pf.make_bool("Inverse", rp.inverse)
        self.addChild(self.inverse)

        # Dense toggle
        # ==================================================
        self.dense = pf.make_bool("Dense", rp.inverse)
        self.addChild(self.inverse)

        # Feedback toggle
        # ==================================================
        self.create_feedback = pf.make_bool("Create feedback", rp.create_feedback)
        self.addChild(self.create_feedback)

        # Kernel parameters
        # ==================================================
        p = {
            "type": "kernel_params",
            "name": "kernel_params",
            "title": "Kernel parameters",
            "kernel_params": kernel_params,
        }

        self.kernel_params = KernelParameterGroup(**p)
        self.addChild(self.kernel_params)


registerParameterType("rf_params", RFParameterGroup)
