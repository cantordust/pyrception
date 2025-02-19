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
        shape: tuple[int, ...],
        bipolar: BipolarLayer,
        sectors: int = 32,
        name: str = "Amacrine",
        rf_params: dict[str, tp.Any] = None,
        notifier: tp.Callable = None,
    ):

        # Initialise the base
        super().__init__(shape, name, notifier)

        # Store the bipolar layer
        self.bipolar = bipolar

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Receptive fields")
        self.rfs = ReceptiveFields(
            self.shape,
            bipolar.rfs.cell_coordinates,
            sectors,
            notifier=notifier,
            **rf_params,
        )
        self.rfs.make_rfs()

        # Membrane potential
        self.membrane = np.zeros((self.rfs.neuron_count,))

        self.info("Initialised.")

    def forward(
        self,
        dt: float | None = None,
    ) -> np.ndarray:

        # Compute the activation of the amacrine cells
        self.membrane = self.convolve(self.rfs.forward_synapses, self.bipolar.membrane)

        return self.membrane

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("rf_colour", "#ff00ffff")
        return self._plot_rfs(self.rfs, *args, **kwargs)
