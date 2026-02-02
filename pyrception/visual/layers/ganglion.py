import numpy as np

import typing as tp

from pyrception.utils import plot
from bokeh.plotting import figure
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import LayerBase
from pathlib import Path
from pyrception.utils.processors import VideoLoader
from pyrception.utils.processors import VideoRecorder


class GanglionLayer(LayerBase):
    """
    A layer of ganglion cells receiving excitatory input from
    bipolar cells and inhibitory input from amacrine cells.
    """

    def __init__(
        self,
        center: ReceptiveFields,
        surround: ReceptiveFields,
        tau: float = 5.0,
        threshold: float = 1e-3,
        inhibition_strength: float = 1.25,
        *args,
        **kwargs,
    ):
        """
        Ganglion layer with center-surround receptive fields.

        Args:

            center: The center receptive fields providing excitatory input to the ganglion cells.
            surround: The surround receptive fields providing inhibitory input to the ganglion cells.
            tau: Membrane time constant.
            threshold: Spiking threshold.
            inhibition_strength: Strength of the inhibitory signal relative to the excitatory one.
        """

        # Initialise the base
        super().__init__(center.size, *args, **kwargs)

        self.center = center
        self.surround = surround

        self.tau = tau
        self.tau_inv = 1 / tau
        self.threshold = threshold
        self.inhibition_strength = inhibition_strength
        self.membrane = np.zeros((self.center.cell_count,))
        self.spikes = np.zeros_like(self.membrane)
        self.logger.info("Initialised.")

    def _spike_frame(self):
        self._canvas *= 0
        self._canvas[
            self.center.cell_coordinates[:, 0],
            self.center.cell_coordinates[:, 1],
        ] = self.spikes
        return self._canvas

    def update_state(
        self,
        current: np.ndarray,
        dt: float | None = None,
    ):
        self.membrane *= 0.0 if dt is None else np.exp(-self.tau_inv * dt)
        self.membrane += current

    def forward(
        self,
        center: np.ndarray,
        surround: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Compute the activation of center/surround RGCs.

        This is where spikes are produced.

        Args:
            center: Excitatory input to the center.
            surround: Inhibitory input to the surround.
            dt: Indicates the time since the last input in the case of temporal integration.

        Returns:
            A spike array.
        """

        # ON centre / OFF surround
        current = self.convolve(
            self.center.forward_synapses, center
        ) - self.inhibition_strength * self.convolve(
            self.surround.forward_synapses, surround
        )

        self.update_state(current, dt)

        self.spikes = np.where(self.membrane >= self.threshold, 1, 0)

        for recorder in self._recorders.values():
            recorder.update()

        return self.spikes

    def plot_activations(self, canvas: np.ndarray = None) -> np.ndarray:

        if canvas is None:
            canvas = np.zeros((self.center.height, self.center.width))
        canvas[:] *= 0.0
        canvas[
            self.center.cell_coordinates[:, 0],
            self.center.cell_coordinates[:, 1],
        ] = self.spikes

        return canvas

    def visualise_spikes(self) -> figure:
        """
        Visualise the activations of amacrine cells.

        Returns:
            A Bokeh figure.
        """
        img = plot.image(self.plot_activations(), title="Ganglion layer | Spikes")
        return plot.show_composite([[img]])

    def record_spikes(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "spikes",
            fpath,
            self._spike_frame,
            self.center.size,
            vl=vl,
        )
