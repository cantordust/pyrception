# --------------------------------------
import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.rf import ReceptiveFields


class AmacrineLayer(BaseLayer):

    def __init__(
        self,
        shape: tp.Tuple[int, ...],
        bipolar: BipolarLayer,
        sectors: int = 32,
        name: str = "Amacrine",
        rf_params: tp.Dict[str, tp.Any] = None,
    ):

        # Initialise the base
        super().__init__(shape, name)

        # Store the bipolar layer
        self.bipolar = bipolar

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Amacrine RFs")
        self.rfs = ReceptiveFields(
            self.shape,
            bipolar.rfs.cell_coordinates,
            sectors,
            **rf_params,
        )
        self.rfs.make_rfs()

        # Activations for the ON and OFF pathways
        self.on = np.zeros((self.rfs.neuron_count,))
        self.off = np.zeros_like(self.on)

        self.info("Initialised.")

    def forward(self):

        # Compute the activation of the amacrine cells
        self.on = self.convolve(self.rfs.forward_synapses, self.bipolar.on)
        self.off = self.convolve(self.rfs.forward_synapses, self.bipolar.off)

        return (self.on, self.off)

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("title", "Amacrine layer receptive fields")
        kwargs.setdefault("rf_colour", "#ff00ffff")
        return self._plot_rfs(self.rfs, *args, **kwargs)
