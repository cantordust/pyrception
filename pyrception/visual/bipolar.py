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
import torch.nn.functional as ptf

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import (
    View,
    RFSizeDist,
    RFType,
    RFType,
)
from pyrception.visual.proto import ProtoLayer
from pyrception.visual.receptor import ReceptorLayer


class BipolarLayer(ProtoLayer):

    """
    A field of bipolar cells.
    This layer processes the signal form the receptor layer
    and passes it on to the RGC layer.
    """

    def __init__(
        self,
        source: ReceptorLayer,
        saccades: bool,
        alpha: float = 0.1,
        k_min: int = 1,
        k_max: int = 11,
        mh: Optional[int] = None,
        mw: Optional[int] = None,
        sh: int = 1 / 8,
        sw: int = 1 / 8,
        rfsizedist: RFSizeDist = RFSizeDist.Gaussian,
        rftype: RFType = RFType.CentreSurround,
        decreasing: bool = False,
        smooth: bool = True,
        layer_name: str = "Bipolar",
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

        self.alpha = alpha

        # Temporal mean
        self.tmean = pt.zeros(
            self.dim.H,
            self.dim.W,
        )

        (self.kmask, ksizes) = self.get_kdist(
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

        # Receptive fields.
        self.rf = self._make_rf(
            ksizes,
            rftype=rftype,
        )

        logger.info(f"==[ {layer_name:<8s} ] Initialisation complete.")

    @logger.catch
    def process(
        self,
        views: Dict[View, pt.Tensor],
        n_frame: int,
        save_frames: Set[int],
        save_views: Set[View],
        frame_paths: Optional[Dict[View, Path]],
    ) -> pt.Tensor:
        """
        Compute the offset from the running mean
        in both positive and negative direction.

        These define the ON and OFF channels.
        """

        _views = {}

        frame = self._convolve(
            views[View.ReceptorAdapted],
            self.rf,
            self.dim.H,
            self.dim.W,
        )
        diff = frame - self.tmean

        _views[View.BipolarOn] = ptf.relu(diff)
        _views[View.BipolarOff] = ptf.relu(-diff)
        _views[View.BipolarCombined] = diff

        # Update the running mean
        self.tmean += self.alpha * diff

        _views[View.BipolarMean] = self.tmean

        if n_frame in save_frames:
            self._save_views(_views, n_frame, save_views, frame_paths)

        views.update(_views)

        # print(f"==[ self.tmean:\n{self.tmean} ({self.tmean.shape})")
