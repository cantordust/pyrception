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
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelType
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.layers.amacrine import AmacrineLayer
from pyrception.visual.layers.proto import ProtoLayer


class GanglionLayer(ProtoLayer):
    """
    A layer of ganglion cells receiving input from bipolar and amacrine cells.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Ganglion")
        super().__init__(*args, **kwargs)
        self.info("Initialised.")

        self.feedback = pt.zeros(
            (self.h, self.w),
            dtype=conf.dtype,
        )

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

        on_center = self.convolve(on, self.rf_centre, self.dim.H, self.dim.W)
        off_center = self.convolve(off, self.rf_centre, self.dim.H, self.dim.W)

        on_surround = self.convolve(on, self.rf_surround, self.dim.H, self.dim.W)
        off_surround = self.convolve(off, self.rf_surround, self.dim.H, self.dim.W)

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
