import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.receptor import ReceptorLayer
from pyrception.visual.rf import ReceptiveFields


class HorizontalLayer(BaseLayer):

    def __init__(
        self,
        shape: tp.Tuple[int, ...],
        receptor: ReceptorLayer,
        sectors: int = 32,
        name: str = "Horizontal",
        rf_params: tp.Dict[str, tp.Any] = None,
        notifier: tp.Callable = None,
    ):

        # Initialise the base
        super().__init__(shape, name, notifier)
        self.receptor = receptor

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Receptive fields")
        rf_params.setdefault("create_feedback", True)
        self.rfs = ReceptiveFields(
            self.shape,
            receptor.rfs.cell_coordinates,
            sectors,
            notifier=notifier,
            **rf_params,
        )
        self.rfs.make_rfs()

        # The vector of horizontal cell activations.
        self.activation = None

        # Feedback matrix.
        self.feedback = None

        self.info("Initialised.")

    def forward(self):

        # Compute the activation of the horizontal cells
        self.activation = self.convolve(self.rfs.forward_synapses, self.receptor.activation)

        # The feedback signal (spatial mean) fed
        # back to the receptors.
        # ==================================================
        self.feedback = self.convolve(self.rfs.feedback_synapses, self.activation)

        return (self.activation, self.feedback)

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("title", "Horizontal layer receptive fields")
        return self._plot_rfs(self.rfs, *args, **kwargs)