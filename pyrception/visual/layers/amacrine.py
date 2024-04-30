from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception import conf
from pyrception.visual.layers.proto import ProtoLayer


class AmacrineLayer(ProtoLayer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Amacrine")
        super().__init__(*args, **kwargs)
        self.info("Initialised.")

    def process(
        self,
        x: pt.Tensor,
        dt: float,
    ):

        # Compute the activation of the amacrine cells
        out_new = self.convolve(x)



        return out_new
