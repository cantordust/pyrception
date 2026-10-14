import numpy as np
import skimage as ski
from pathlib import Path

from bokeh.plotting import figure
from bokeh.io.notebook import CommsHandle

from pyrception.utils import plot
from pyrception.visual import ReceptiveFields
from pyrception.visual.layers.base import LayerBase
from pyrception.utils.processors import VideoLoader
from pyrception.utils.processors import VideoRecorder


class HorizontalLayer(LayerBase):

    def __init__(
        self,
        excitatory: ReceptiveFields,
        *args,
        **kwargs,
    ):
        super().__init__(excitatory.size, *args, **kwargs)

        self.excitatory = excitatory

        self.raw = np.zeros(self._size)
        self.feedback = np.zeros(self._size)

        self.activations: np.ndarray | None = None
        self.feedback: np.ndarray | None = None
        self.logger.info("Initialised.")

    def activation_frame(self, canvas: np.ndarray) -> np.ndarray:
        """
        A frame showing the neuron activations.

        Returns:
            A frame as a NumPy array.
        """
        canvas[:] = 0
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = ski.exposure.rescale_intensity(self.activations, out_range=np.ubyte)
        return canvas

    def feedback_frame(self, canvas: np.ndarray) -> np.ndarray:
        """
        A frame showing the feedback response.

        Returns:
            A frame as a NumPy array.
        """
        canvas[:] = ski.exposure.rescale_intensity(
            self.feedback.reshape(self.excitatory.size),
            out_range=np.ubyte,
        )
        return canvas

    def norm_frame(self, canvas: np.ndarray) -> np.ndarray:
        """
        Frame showing the normalised input (raw - feedback).

        Returns:
            A NumPy array.
        """
        canvas[:] = (self.raw - self.feedback).reshape(self.excitatory.size[:2])
        return canvas

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, ...]:

        self.raw = x

        # Forward and feedback signals
        self.activations = self.convolve(self.excitatory.forward_synapses, x)
        self.feedback = self.convolve(
            self.excitatory.feedback_synapses, self.activations
        ).flatten()

        for recorder in self._recorders.values():
            recorder.update()

        return (self.activations, self.feedback)

    def visualise_activations(self):
        return plot.show_composite(plot.image(self._activation_frame()))

    def visualise_feedback(self):
        return plot.show_composite(
            plot.image(self.feedback.reshape(self.excitatory.size))
        )

    def visualise_norm_input(
        self,
        signal: np.ndarray,
    ) -> CommsHandle | None:
        """
        Visualise the input normalised by the feedback of the horizontal cells.

        Args:
            signal: The input signal.

        Returns:
            The normalised input.
        """
        normalised_canvas = signal - self.feedback.reshape(
            (self.excitatory.height, self.excitatory.width)
        )
        normalised_canvas = ski.exposure.rescale_intensity(
            normalised_canvas, out_range=(0.0, 1.0)
        )

        scaled_plot = plot.image(normalised_canvas, title="Normalised input")

        return plot.show_composite(scaled_plot)

    def plot_activations(
        self,
        xtitle: str = "Neuron",
        ytitle: str = "Activation",
        width: int = 1000,
    ) -> figure:
        """
        Return a scatter plot of the activations of this layer's neurons.

        Args:
            xtitle: Title for the x axis.
            ytitle: Title for the y axis.
            width: Plot width.

        Returns:
            A Bokeh figure.
        """
        scatter = plot.scatter(
            self.activations,
            xtitle=xtitle,
            ytitle=ytitle,
            width=width,
        )
        return plot.show_composite(scatter)

    def record_activations(
        self,
        fpath: str | Path = "./horizontal_activations.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        """
        Record the neuron activations.

        Args:
            fpath: The file to save the recording to.
            vl: An optional VideoLoader instance.

        Returns:
            A VideoRecorder instance.
        """
        return self.record(
            "activations",
            fpath,
            self._activation_frame,
            # size=self.excitatory.size,
            vl=vl,
        )

    def record_feedback(
        self,
        fpath: str | Path = "./horizontal_feedback.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        """
        Record the feedback signal from the neurons.

        Args:
            fpath: The file to save the recording to.
            vl: An optional VideoLoader instance.

        Returns:
            A VideoRecorder instance.
        """
        return self.record(
            "feedback",
            fpath,
            self._feedback_frame,
            # size=self.excitatory.size,
            vl=vl,
        )

    def record_norm(
        self,
        fpath: str | Path = "./horizontal_norm.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.record(
            "norm",
            fpath,
            self._norm_frame,
            # size=self.excitatory.size,
            vl=vl,
        )
