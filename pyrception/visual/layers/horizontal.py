from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception.visual.layers.proto import ProtoLayer


class HorizontalLayer(ProtoLayer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Horizontal")
        super().__init__(*args, **kwargs)
        self.log("Initialised.")

        # Exponential running mean of the input
        # ==================================================
        self.mean = pt.zeros((self.neuron_count,), dtype=pt.float32)
        self.feedback = pt.zeros(
            (self.h, self.w),
            dtype=pt.float32,
        )

    def process(
        self,
        frame: pt.Tensor,
        dt: float,
    ):

        # Compute the activation of the horizontal cells
        out_new = self._convolve(frame)

        # Update the running mean
        self.mean += self.alpha * (out_new - self.mean)

        # Update the feedback tensor
        self.feedback.zero_()
        for idx, (rows, cols) in enumerate(zip(self.rows, self.cols)):
            self.feedback[rows, cols] += self.mean[idx]

        # Scale feedback from overlapping horizontal cells
        self.feedback *= self.rf_factors

        # Normalise the frame using the feedback tensor
        norm = frame - self.feedback

        return norm
