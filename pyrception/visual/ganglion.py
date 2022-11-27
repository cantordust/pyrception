from typing import Set
from typing import Dict
from typing import Tuple
from typing import Optional

# --------------------------------------
from loguru import logger

# --------------------------------------
from pathlib import Path

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
        saccades: bool,
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
        layer_name: str = "Ganglion",
    ):

        logger.info(f"==[ {layer_name:<8s} ] Initialising layer...")

        self.source = source

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(
            source.dim.shape,
            source.dim.H,
            source.dim.W,
            saccades,
        )

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

        surround_ksizes = (centre_ksizes * 2 + 1).int()
        # surround_ksizes = centre_ksizes

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

        logger.info(f"==[ {layer_name:<8s} ] Initialisation complete.")

    def process(
        self,
        views: Dict[View, pt.Tensor],
        n_frame: int,
        save_frames: Set[int],
        save_views: Set[View],
        frame_paths: Optional[Dict[View, Path]],
    ) -> pt.Tensor:
        """
        Compute the activation of ON/OFF and OFF/ON RGCs.

        This is where spikes are produced.
        """

        _views = {}

        on = views[View.BipolarOn]
        off = views[View.BipolarOff]

        on_center = self._convolve(on, self.rf_centre, self.dim.H, self.dim.W)
        off_center = self._convolve(off, self.rf_centre, self.dim.H, self.dim.W)

        on_surround = self._convolve(on, self.rf_surround, self.dim.H, self.dim.W)
        off_surround = self._convolve(off, self.rf_surround, self.dim.H, self.dim.W)

        threshold = 15

        _views[View.GanglionOnOff] = pt.where(
            on_center - off_surround > threshold, 1.0, 0.0
        )
        _views[View.GanglionOffOn] = pt.where(
            off_center - on_surround > threshold, 1.0, 0.0
        )
        _views[View.OnOffEvents] = on - off

        if n_frame in save_frames:
            self._save_views(_views, n_frame, save_views, frame_paths)

        views.update(_views)
