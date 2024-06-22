from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.rf import ReceptiveFields


class AmacrineLayer(BaseLayer):

    def __init__(
        self,
        size: Tuple[int, ...],
        bipolar: BipolarLayer,
        sectors: int = 32,
        name: str = "Amacrine",
        rf_params: Dict[str, Any] = None,
    ):

        # Initialise the base
        super().__init__(size, name)

        # Store the bipolar layer
        self.bipolar = bipolar

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Amacrine RFs")
        self.rfs = ReceptiveFields(
            self.size,
            bipolar.rfs.cell_coordinates,
            sectors,
            **rf_params,
        )
        self.rfs.make_rfs()

        # Activations for the ON and OFF pathways
        self.on = pt.zeros((self.rfs.neuron_count,))
        self.off = pt.zeros_like(self.on)

        self.info("Initialised.")

    def forward(self):

        # Compute the activation of the amacrine cells
        self.on = self.convolve(self.rfs.rfs, self.bipolar.on)
        self.off = self.convolve(self.rfs.rfs, self.bipolar.off)

        return (self.on, self.off)

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("title", "Amacrine layer receptive fields")
        kwargs.setdefault("rf_colour", "#ff00ffff")
        return self._plot_rfs(self.rfs, *args, **kwargs)
