# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Tuple
from typing import Optional

# --------------------------------------
from loguru import logger

# --------------------------------------
import torch as pt
import torch.functional as ptf

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import (
    View,
    RFSizeDist,
    RFType,
    RFType,
)
from pyrception.visual.bipolar import BipolarLayer
from pyrception.visual.proto import ProtoLayer


class GanglionLayer(ProtoLayer):

    """
    A layer of ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        source: BipolarLayer,
        saccades: bool = False,
        k_min: int = 1,
        k_max: int = 9,
        mh: Optional[int] = None,
        mw: Optional[int] = None,
        sh: int = 1 / 8,
        sw: int = 1 / 8,
        rfsizedist: RFSizeDist = RFSizeDist.Gaussian,
        rftype: RFType = RFType.CentreSurround,
        decreasing: bool = False,
        smooth: bool = True,
    ):

        # Ganglion cells.
        self.source = source

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(
            source.dim.shape,
            source.dim.H,
            source.dim.W,
            saccades,
        )

        logger.info(f"==[ ganglion ] dim: {self.dim}")

        (_, centre_ksizes) = self.get_kdist(
            self.dim.H,
            self.dim.W,
            k_min,
            k_max,
            mh,
            mw,
            sh,
            sw,
            rfsizedist,
            decreasing,
            smooth,
        )

        surround_ksizes = (centre_ksizes * 2 - 1).int()

        # Receptive fields.
        self.rf_centre = self._make_rf(
            centre_ksizes,
            rftype=rftype,
            norm=True,
        )

        self.rf_surround = self._make_rf(
            surround_ksizes,
            rftype=rftype,
            norm=True,
        )

    def process(
        self,
        on: pt.Tensor,
        off: pt.Tensor,
    ) -> pt.Tensor:
        """
        Compute the activation of ON/OFF and OFF/ON RGCs.

        This is where spikes are produced.
        """

        views = {}

        on_center = self._convolve(on, self.rf_centre, self.dim.H, self.dim.W)
        off_center = self._convolve(off, self.rf_centre, self.dim.H, self.dim.W)

        on_surround = self._convolve(on, self.rf_surround, self.dim.H, self.dim.W)
        off_surround = self._convolve(off, self.rf_surround, self.dim.H, self.dim.W)

        views[View.GanglionOnOff] = pt.where(on_center - off_surround > 1, 1.0, 0.0)
        views[View.GanglionOffOn] = pt.where(off_center - on_surround > 1, 1.0, 0.0)
        views[View.OpticalFlow] = on_center - off_surround

        return views
