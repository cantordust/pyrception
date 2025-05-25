import numpy as np

import typing as tp

from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.layers.amacrine import AmacrineLayer


class GanglionLayer(BaseLayer):
    """
    A layer of ganglion cells receiving excitatory input from
    bipolar cells and inhibitory input from amacrine cells.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        bipolar: BipolarLayer,
        amacrine: AmacrineLayer,
        sectors: int = 64,
        name: str = "Ganglion",
        inhibition_scale: float = 1,
        bipolar_params: dict[str, tp.Any] = None,
        amacrine_params: dict[str, tp.Any] = None,
        tau: float | np.ndarray = 1.0,
        threshold: float | np.ndarray = 0.5,
        notifier: tp.Callable = None,
    ):
        """
        Generic ganglion layer.

        Args:
            shape:
                Input shape.

            bipolar:
                The bipolar layer providing excitatory input to the ganglion cells.

            amacrine:
                The amacrine layer providing inhibitory input to the ganglion cells.

            sectors:
                Number of sectors (angular sections).

            name:
                The name of the layer.

            inhibition_scale:
                Strength of the inhibitory signal relative to the excitatory one.

            bipolar_params:
                Parameters for the bipolar RFs.

            amacrine_params:
                Parameters for the amacrine RFs.

            tau:
                Membrane time constant.

            threshold:
                Spiking threshold.

            notifier:
                An optional function for progress notification and messaging.

        """

        # Initialise the base
        super().__init__(shape, name, notifier)

        # Initialise the bipolar receptive fields.
        # ==================================================
        self.bipolar = bipolar
        if bipolar_params is None:
            bipolar_params = {}
        bipolar_params.setdefault("name", f"{name} | Bipolar receptive fields")
        bipolar_params["sectors"] = sectors
        self.bipolar_rfs = ReceptiveFields(
            self.shape,
            bipolar.rfs.cell_coordinates,
            notifier=notifier,
            **bipolar_params,
        )
        self.bipolar_rfs.make_rfs()

        # Initialise the amacrine receptive fields.
        # ==================================================
        self.amacrine = amacrine
        if amacrine_params is None:
            amacrine_params = {}
        amacrine_params.setdefault("name", f"{name} | Amacrine receptive fields")
        amacrine_params["sectors"] = sectors
        self.amacrine_rfs = ReceptiveFields(
            self.shape,
            amacrine.rfs.cell_coordinates,
            **amacrine_params,
        )
        self.amacrine_rfs.make_rfs()
        self.inhibition_scale = inhibition_scale

        self.tau = tau
        self.tau_inv = 1 / tau
        self.membrane = np.zeros((self.bipolar_rfs.neuron_count,))
        self.spikes = np.zeros_like(self.membrane)
        self.threshold = threshold

        self.info("Initialised.")

    def update_state(
        self,
        current: np.ndarray,
        dt: float | None = None,
    ):
        self.membrane *= (
            0.0 if dt is None else (self.tau_inv * np.exp(-self.tau_inv * dt))
        )
        self.membrane += current

    def forward(
        self,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Compute the activation of center/surround RGCs.

        This is where spikes are produced.

        Args:

            dt: Indicates the time since the last input in the case of temporal integration.

        Returns:
            A spike array.
        """

        # ON centre / OFF surround
        current = self.convolve(
            self.bipolar_rfs.forward_synapses,
            self.bipolar.membrane,
        ) - self.inhibition_scale * self.convolve(
            self.amacrine_rfs.forward_synapses,
            self.amacrine.membrane,
        )

        self.update_state(current, dt)

        self.spikes = np.where(self.membrane >= self.threshold, 1, 0)

        return self.spikes

    def plot_rfs(
        self,
        bipolar_rf_colour: str = "#ff00ff",
        amacrine_rf_colour: str = "#ffff00",
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Plot the receptive fields of amacrine cells.
        This takes into account the sparsity of bipolar cells.

        Args:

            bipolar_rf_colour: The colour to use for highlighting the plotted bipolar cells.
            amacrine_rf_colour: The colour to use for highlighting the plotted amacrine cells.

        Returns:
            A plot of the receptive fields of the cells.
        """

        # Plot the bipolar cells
        bl_canvas = self._plot_rfs(
            self.bipolar_rfs,
            rf_colour=bipolar_rf_colour,
            *args,
            **kwargs,
        )

        # Plot the amacrine cells
        al_canvas = self._plot_rfs(
            self.amacrine_rfs,
            rf_colour=amacrine_rf_colour,
            *args,
            **kwargs,
        )

        canvas = bl_canvas + al_canvas
        canvas /= canvas.max()

        return canvas
