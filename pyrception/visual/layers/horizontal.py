from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from pyrception import conf
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.receptor import ReceptorLayer
from pyrception.visual.rf import ReceptiveFields


class HorizontalLayer(BaseLayer):

    def __init__(
        self,
        size: Tuple[int, ...],
        receptor: ReceptorLayer,
        sectors: int = 32,
        name: str = "Horizontal",
        rf_params: Dict[str, Any] = None,
    ):

        # Initialise the base
        super().__init__(size, name)
        self.receptor = receptor

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Horizontal RFs")
        self.rfs = ReceptiveFields(
            self.size,
            receptor.rfs.cell_coordinates,
            sectors,
            compute_factors=True,
            **rf_params,
        )
        self.rfs.make_rfs()

        # The vector of horizontal cell activations.
        self.activation = None

        # Feedback matrix.
        self.feedback = pt.zeros(
            (self.rfs.height * self.rfs.width,),
            dtype=conf.dtype,
            device=conf.device,
        )

        # Indices
        self.feedback_indices = [
            cols + self.rfs.width * rows
            for rows, cols in zip(self.rfs.rows, self.rfs.cols)
        ]

        self.info("Initialised.")

    def forward(self):

        # Compute the activation of the horizontal cells
        self.activation = self.convolve(
            self.rfs.rfs,
            self.receptor.activation,
        )

        # The feedback signal (spatial mean) fed
        # back to the receptors.
        # ==================================================
        self.feedback.zero_()
        for idx in range(len(self.activation)):
            self.feedback[self.feedback_indices[idx]] += self.activation[idx]

        # Scale the feedback for overlapping dendritic fields
        self.feedback *= self.rfs.rf_factors

        return (self.activation, self.feedback)

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("title", "Horizontal layer receptive fields")
        return self._plot_rfs(self.rfs, *args, **kwargs)
