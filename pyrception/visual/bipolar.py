# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Tuple
from typing import Optional

# --------------------------------------
from loguru import logger

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelSizeDist
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
        saccades: bool = False,
        alpha: float = 0.1,
        k_min: int = 1,
        k_max: int = 11,
        mh: Optional[int] = None,
        mw: Optional[int] = None,
        sh: int = 1 / 8,
        sw: int = 1 / 8,
        kdist: KernelSizeDist = KernelSizeDist.Gaussian,
        decreasing: bool = False,
        smooth: bool = True,
    ):

        self.source = source

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(
            source.dim.shape,
            source.dim.H,
            source.dim.W,
            saccades,
        )

        logger.info(f"==[ bipolar ] dim : {self.dim}")

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
            kdist,
            decreasing,
            smooth,
        )

        # Receptor fields.
        self.rf = self._make_rf(ksizes)

    @logger.catch
    def process(
        self,
        frame: pt.Tensor,
        saccades: Optional[Tuple[float, float]] = None,
    ) -> pt.Tensor:
        """
        Compute the offset from the running mean
        in both positive and negative direction.

        These define the ON and OFF channels.
        """

        # Update the internal state

        views = {
            View.BipolarMean: self.tmean,
        }

        frame = self._convolve(frame, self.rf, self.dim.H, self.dim.W)
        diff = frame - self.tmean

        views[View.BipolarOn] = ptf.relu(diff)
        views[View.BipolarOff] = ptf.relu(-diff)

        # Update the running mean
        self.tmean += self.alpha * diff

        # print(f"==[ self.tmean:\n{self.tmean} ({self.tmean.shape})")

        return views
