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
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelType
from pyrception.visual.layers.horizontal import HorizontalLayer
from pyrception.visual.layers.proto import ProtoLayer


class BipolarLayer(ProtoLayer):
    """
    A layer of bipolar cells.
    This layer processes the signal form the receptor layer
    modulated by the horizontal layer.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Bipolar")
        super().__init__(*args, **kwargs)

        # Temporal mean
        self.tmean = pt.zeros((self.neuron_count,))
        self.on = pt.zeros((self.neuron_count,))
        self.off = pt.zeros_like(self.on)

        self.log("Initialised.")

    def process(
        self,
        frame: int,
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

        if frame in save_frames:
            self._save_views(_views, frame, save_views, frame_paths)

        views.update(_views)

        # print(f"==[ self.tmean:\n{self.tmean} ({self.tmean.shape})")
