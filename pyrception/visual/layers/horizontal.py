from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception import conf
from pyrception.visual.layers.proto import ProtoLayer


class HorizontalLayer(ProtoLayer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Horizontal")
        super().__init__(*args, **kwargs)
        self.info("Initialised.")

    def __call__(
        self,
        raw: pt.Tensor,
    ):

        # Compute the activation of the horizontal cells
        activation = self.convolve(raw)

        # The feedback signal (spatial mean) fed
        # back to the receptors.
        # ==================================================
        feedback = pt.zeros((self.h, self.w), dtype=conf.dtype)
        for idx, (rows, cols) in enumerate(zip(self.rows, self.cols)):
            feedback[rows, cols] += activation[idx]

        # # Scale the feedback for overlapping dendritic fields
        feedback *= self.rf_factors

        return (activation, feedback)
