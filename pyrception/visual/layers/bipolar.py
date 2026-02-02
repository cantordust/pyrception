import numpy as np
from pyrception.utils import plot
from pathlib import Path
from bokeh.plotting import figure
import skimage as ski
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import LayerBase
from pyrception.utils.processors import VideoLoader
from pyrception.utils.processors import VideoRecorder


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

    def _on_frame(self):
        self._canvas *= 0
        self._canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = self.on
        return self._canvas

    def _off_frame(self):
        self._canvas *= 0
        self._canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = self.off
        return self._canvas

    def _diff_frame(self):
        self._canvas *= 0
        self._canvas[
            self.excitatory.cell_coordinates[:, 0],
            self.excitatory.cell_coordinates[:, 1],
        ] = (
            self.on - self.off
        )
        return self._canvas

    def _on_off_frame(self):
        """
        _summary_

        Returns:
            _description_
        """
        canvas = np.zeros((*self._size, 3))
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

    def _ring_delog_canvas_on(self):
        """
        _summary_

        Returns:
            _description_
        """
        max_ring_size = np.max([ring.size for ring in self.excitatory.cell_rings])
        ring_array = np.zeros((len(self.excitatory.cell_rings), max_ring_size))
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
            ring_array[r, max_ring_size // 2 - len(left_arc) : max_ring_size // 2] = (
                self.on[left_arc]
            )
            ring_array[r, max_ring_size // 2 : max_ring_size // 2 + len(right_arc)] = (
                self.on[right_arc]
            )

        return ring_array

    def _sector_delog_canvas_on(self):
        """
        _summary_

        Returns:
            _description_
        """
        max_sector_size = np.max([ring.size for ring in self.excitatory.cell_rings])
        sector_array = np.zeros((len(self.excitatory.cell_sectors), max_sector_size))
        for s in range(len(self.excitatory.cell_sectors)):
            sector_array[s, : len(self.excitatory.cell_sectors[s])] = self.on[
                self.excitatory.cell_sectors[s]
            ]

        return sector_array

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
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "on",
            fpath,
            self._on_frame,
            self.excitatory.size,
            vl=vl,
        )

    def record_off(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "off",
            fpath,
            self._off_frame,
            self.excitatory.size,
            vl=vl,
        )

    def record_on_off(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "on_off",
            fpath,
            self._on_off_frame,
            self.excitatory.size,
            vl=vl,
        )

    def record_diff(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        return self.add_recorder(
            "diff",
            fpath,
            self._diff_frame,
            self.excitatory.size,
            vl=vl,
        )

    def record_ring_delog_on(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:

        max_ring_size = np.max([ring.size for ring in self.excitatory.cell_rings])
        ring_array = np.zeros((len(self.excitatory.cell_rings), max_ring_size))
        return self.add_recorder(
            "ring_delog_on",
            fpath,
            self._ring_delog_canvas_on,
            ring_array.shape,
            vl=vl,
        )

    def record_sector_delog_on(
        self,
        fpath: Path,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:

        max_sector_size = np.max([ring.size for ring in self.excitatory.cell_rings])
        sector_array = np.zeros((len(self.excitatory.cell_rings), max_sector_size))
        return self.add_recorder(
            "sector_delog_on",
            fpath,
            self._sector_delog_canvas_on,
            sector_array.shape,
            vl=vl,
        )
