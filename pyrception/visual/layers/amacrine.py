import numpy as np

from pyrception.utils import plot
from bokeh.plotting import figure
from pathlib import Path
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import LayerBase
from pyrception.utils.processors import VideoLoader
from pyrception.utils.processors import VideoRecorder


class AmacrineLayer(LayerBase):
    def __init__(
        self,
        excitatory: ReceptiveFields,
        *args,
        **kwargs,
    ):
        # Initialise the base
        super().__init__(excitatory.size, *args, **kwargs)

        # Store the bipolar layer
        self.excitatory = excitatory

        # Membrane potential
        self.activations = np.zeros((self.excitatory.cell_count,))

        self.logger.info("Initialised.")

    def _activation_frame(self):
        self._canvas *= 0
        self._canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = self.activations
        return self._canvas

    def forward(
        self,
        bipolar: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:

        self.activations = self.convolve(self.excitatory.forward_synapses, bipolar)

        for recorder in self._recorders.values():
            recorder.update()
        return self.activations

    def visualise_activations(self) -> figure:
        """
        Visualise the activations of amacrine cells.

        Returns:
            A Bokeh figure.
        """

        canvas = np.zeros((self.excitatory.height, self.excitatory.width))
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = self.activations
        img = plot.image(canvas, title="Amacrine layer | Activations")
        return plot.show_composite([[img]])

    def record_activations(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "activations",
            fpath,
            self._activation_frame,
            self.excitatory.size,
            vl=vl,
        )
