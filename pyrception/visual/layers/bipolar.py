import numpy as np
from pyrception.utils import plot
from pathlib import Path
from bokeh.plotting import figure
import skimage as ski
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import LayerBase
from pyrception.utils.processors import VideoLoader, VideoRecorder


class BipolarLayer(LayerBase):

    def __init__(
        self,
        excitatory: ReceptiveFields,
        alpha: int | float | tuple[float, float] = (0.05, 0.25),
        *args,
        **kwargs,
    ):
        """
        A layer of bipolar cells.
        This layer processes the signal form the receptor layer
        modulated by the horizontal layer.
        """
        # Initialise the base
        super().__init__(excitatory.size, *args, **kwargs)

        self.excitatory = excitatory

        # Exponential running mean decay rate
        self.alpha = self._compute_alpha(alpha)

        # Membrane potential
        self.membrane = np.zeros((self.excitatory.cell_count,))
        self.on = np.zeros_like(self.membrane)
        self.off = np.zeros_like(self.membrane)
        self.norm = np.zeros(self.excitatory.size[:2])

        self.logger.info("Initialised.")

    def _compute_alpha(
        self,
        alpha: int | float | tuple[float, float] = (0.05, 0.25),
    ) -> np.ndarray:
        """
        Compute the decay coefficient (alpha) for each neuron.

        Args:
            alpha: The value(s) that alpha could take.

        Returns:
            The distribution of decay coefficients.
        """

        if isinstance(alpha, (int, float)):
            alpha = [alpha, alpha]

        hr = self.excitatory.cell_coordinates[:, 0] - self.excitatory.height // 2
        wr = self.excitatory.cell_coordinates[:, 1] - self.excitatory.width // 2

        distances = np.sqrt(hr**2 + wr**2)
        alpha_min = alpha[0]
        alpha_max = alpha[1]

        alpha = 1 - 1 / (1 + distances / distances.max())

        # Min-max scaling
        alpha = alpha_min + (alpha - alpha.min()) * (alpha_max - alpha_min) / (
            alpha.max() - alpha.min()
        )

        self.logger.debug(f"Alpha range: {alpha.min():>0.3f} - {alpha.max():>0.3f}")

        return alpha

    def on_frame(self, canvas: np.ndarray):
        canvas[:] = 0
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
            :,
        ] = self.on[:, None]
        return canvas

    def off_frame(self, canvas: np.ndarray):
        canvas[:] = 0
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
            :,
        ] = self.off[:, None]
        return canvas

    def diff_frame(self, canvas: np.ndarray):
        canvas[:] = 0
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
            :,
        ] = (
            self.on - self.off
        )[:, None]
        return canvas

    def combined_frame(self, canvas: np.ndarray):
        """
        _summary_

        Returns:
            _description_
        """
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
            0,
        ] = ski.exposure.rescale_intensity(
            self.on,
            out_range=np.ubyte,
        )
        canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
            1,
        ] = ski.exposure.rescale_intensity(
            self.off,
            out_range=np.ubyte,
        )
        return canvas

    def ring_delog_on_frame(self, canvas: np.ndarray, max_ring_size: int):
        """
        _summary_

        Returns:
            _description_
        """

        canvas[:] = 0
        for r in range(len(self.excitatory.cell_rings)):
            left_arc = self.excitatory.cell_rings[r][
                len(self.excitatory.cell_rings[r])
                // 4 : -len(self.excitatory.cell_rings[r])
                // 4
            ]
            right_arc = np.concatenate(
                (
                    self.excitatory.cell_rings[r][
                        -len(self.excitatory.cell_rings[r]) // 4 :
                    ],
                    self.excitatory.cell_rings[r][
                        : len(self.excitatory.cell_rings[r]) // 4
                    ],
                )
            )
            canvas[
                r,
                max_ring_size // 2 - len(left_arc) : max_ring_size // 2,
                :,
            ] = self.on[left_arc][:, None]
            canvas[
                r,
                max_ring_size // 2 : max_ring_size // 2 + len(right_arc),
                :,
            ] = self.on[right_arc][:, None]

        return canvas

    def sector_delog_on_frame(self, canvas: np.ndarray):
        """
        _summary_

        Returns:
            _description_
        """
        canvas[:] = 0
        for s in range(len(self.excitatory.cell_sectors)):
            canvas[
                s,
                : len(self.excitatory.cell_sectors[s]),
                :,
            ] = self.on[self.excitatory.cell_sectors[s]][:, None]

        return canvas

    def forward(
        self,
        raw: np.ndarray,
        feedback: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        The bipolar layer splits the input into ON and OFF pathways.

        Args:
            raw: The raw input signal from the receptor layer.
            feedback: The feedback signal from the horizontal layer.
            dt: (Optional) time since the last input.

        Returns:
            An array with the membrane potentials of the bipolar cells.
        """

        # Take the difference of the raw receptor input and
        # the spatial mean as computed by the horizontal cells.
        self.norm = raw - feedback

        # Compute the nominal activation for each bipolar cell
        activation = self.convolve(self.excitatory.forward_synapses, self.norm)

        # The filter implemented by the bipolar cell is
        # an exponential running mean of the activation.
        # The on/off activations are computed as rectified
        # versions of the deviations from that mean in positive
        # and negative direction, respectively.
        # The mean is effectively filtered out here.
        # ==================================================
        self.membrane += self.alpha * (activation - self.membrane)
        deviation = activation - self.membrane

        self.on = np.where(deviation > 0.0, deviation, 0.0)
        self.off = np.where(deviation < 0.0, -deviation, 0.0)

        for recorder in self._recorders.values():
            recorder.update()

        return (self.on, self.off)

    def visualise_activations(
        self,
        on: bool = True,
        off: bool = True,
    ) -> figure:
        """
        Visualise the activations of ON and OFF bipolar cells.

        Args:
            on: If set, visualise ON cell activations.
            off: If set, visualise ON cell activations.

        Returns:
            A Bokeh figure.
        """

        entries = []
        if on:
            bp_on_canvas = np.zeros((self.excitatory.height, self.excitatory.width))
            bp_on_canvas[
                self.excitatory.cell_coordinates[:, 0],
                self.excitatory.cell_coordinates[:, 1],
            ] = self.on
            entries.append(plot.image(bp_on_canvas, title="ON bipolar"))

        if off:
            bp_off_canvas = np.zeros((self.excitatory.height, self.excitatory.width))
            bp_off_canvas[
                self.excitatory.cell_coordinates[:, 0],
                self.excitatory.cell_coordinates[:, 1],
            ] = self.off
            entries.append(plot.image(bp_off_canvas, title="OFF bipolar"))

        if len(entries) > 0:
            return plot.show_composite([entries])

    def record_on(
        self,
        fpath: str | Path = "./bipolar_on.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.record(
            "on",
            fpath,
            self._on_frame,
            vl=vl,
        )

    def record_off(
        self,
        fpath: str | Path = "./bipolar_off.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.record(
            "off",
            fpath,
            self.off_frame,
            vl=vl,
        )

    def record_combined(
        self,
        fpath: str | Path = "./bipolar_combined.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.record(
            "on_off",
            fpath,
            self.combined_frame,
            vl=vl,
        )

    def record_diff(
        self,
        fpath: str | Path = "./bipolar_diff.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.record(
            "diff",
            fpath,
            self.diff_frame,
            vl=vl,
        )

    def record_ring_delog_on(
        self,
        fpath: str | Path = "./bipolar_ring_delog_on.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:

        max_ring_size = np.max([ring.size for ring in self.excitatory.cell_rings])
        return self.record(
            "ring_delog_on",
            fpath,
            self.ring_delog_on_frame,
            opts={
                "max_ring_size": np.max(
                    [ring.size for ring in self.excitatory.cell_rings]
                )
            },
            size=(len(self.excitatory.cell_rings), max_ring_size),
            codec="ffv1",
            vl=vl,
        )

    def record_sector_delog_on(
        self,
        fpath: str | Path = "./bipolar_sector_delog_on.mp4",
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:

        max_sector_size = np.max([sector.size for sector in self.excitatory.cell_sectors])
        return self.record(
            "sector_delog_on",
            fpath,
            self.sector_delog_on_frame,
            size=(len(self.excitatory.cell_sectors), max_sector_size),
            codec="ffv1",
            vl=vl,
        )
