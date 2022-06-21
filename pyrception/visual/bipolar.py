# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Set
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf

# --------------------------------------
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelDist
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
        *args,
        **kwargs,
    ):

        self.source = source

        # Dimensions and resize flag
        self.dim = self.compute_dimensions(
            source.dim.shape,
            source.dim.H,
            source.dim.W,
            saccades,
        )

        print(f"==[ bipolar ] dim : {self.dim}")

        self.alpha = kwargs.get("alpha", 0.25)

        # Temporal mean
        self.tmean = pt.zeros(
            self.dim.H,
            self.dim.W,
        )

        # Initialise the base
        super().__init__(
            self.dim.H,
            self.dim.W,
            *args,
            **kwargs,
        )

    def get_rf_input(
        self,
        frame: pt.Tensor,
    ) -> pt.Tensor:

        stretched = self.stretch(frame)

        mean = pt.mm(self.rf, stretched)

        folded = self.fold(mean, self.dim.H, self.dim.W)

        return folded

    def process(
        self,
        frame: pt.Tensor,
        saccades: Optional[Tuple[float, float]] = None,
    ) -> pt.Tensor:
        """
        Compute the offset from the running mean
        in both positive and negative direction.

        These become the separate ON and OFF channels.
        """

        # Update the internal state

        views = {
            View.BipolarMean: self.tmean,
        }

        frame = self.get_rf_input(frame)

        diff = frame - self.tmean

        views[View.BipolarOn] = ptf.relu(diff)
        views[View.BipolarOff] = ptf.relu(-diff)

        # Update the running mean
        self.tmean += self.alpha * (frame - self.tmean)

        return views
