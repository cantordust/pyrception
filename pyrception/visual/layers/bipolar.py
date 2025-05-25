import numpy as np

import typing as tp

from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.receptor import ReceptorLayer
from pyrception.visual.layers.horizontal import HorizontalLayer


class BipolarLayer(BaseLayer):
    """
    A layer of bipolar cells.
    This layer processes the signal form the receptor layer
    modulated by the horizontal layer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        receptor: ReceptorLayer,
        horizontal: HorizontalLayer,
        sectors: int = 64,
        name: str = "Bipolar",
        forgetting_range: tuple[float, float] = (0.05, 0.95),
        rf_params: dict[str, tp.Any] = None,
        notifier: tp.Callable = None,
    ):
        # Initialise the base
        super().__init__(shape, name, notifier)

        self.receptor = receptor
        self.horizontal = horizontal

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Receptive fields")
        self.rfs = ReceptiveFields(
            self.shape,
            receptor.rfs.cell_coordinates,
            sectors,
            notifier=notifier,
            **rf_params,
        )
        self.rfs.make_rfs()

        # Range of forgetting rates
        self.forgetting_range = forgetting_range

        # 'Forgetting rate' for the temporal mean
        self.alpha = self.compute_forgetting_rate()

        # Membrane potential
        self.membrane = np.zeros((self.rfs.neuron_count,))

        # Temporal mean (exponential running mean).
        # We initialise this to None because it is initialised
        # with the mean of the first activation map.
        self.mean = None

        self.info("Initialised.")

    def compute_forgetting_rate(self):
        hr = self.rfs.cell_coordinates[:, 0] - self.rfs.height // 2
        wr = self.rfs.cell_coordinates[:, 1] - self.rfs.width // 2

        distances = np.sqrt(hr**2 + wr**2)
        alpha_min = self.forgetting_range[0]
        alpha_max = self.forgetting_range[1]

        alpha = 1 - 1 / (1 + distances / distances.max())

        # Min-max scaling
        alpha = alpha_min + (alpha - alpha.min()) * (alpha_max - alpha_min) / (
            alpha.max() - alpha.min()
        )

        self.debug(f"Forgetting rate range: {alpha.min():>0.3f} - {alpha.max():>0.3f}")

        return alpha

    def _compute_membrane_potential(
        self,
        activation: np.ndarray,
    ):
        self.membrane = np.clip(activation - self.mean, min=0.0)

    def forward(
        self,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        The bipolar layer splits the input into ON and OFF pathways.

        Returns:
            An array with the membrane potentials of the bipolar cells.
        """

        # Take the difference of the raw receptor input and
        # the spatial mean as computed by the horizontal cells.
        scaled_input = self.receptor.membrane - self.horizontal.feedback

        # Compute the nominal activation for each bipolar cell
        activation = self.convolve(self.rfs.forward_synapses, scaled_input)

        # The filter implemented by the bipolar cell is
        # a temporal exponential running mean of the activation.
        # The membrane potential is computed as a rectified
        # version of the deviation from that mean.
        # ==================================================
        if self.mean is None:
            self.mean = activation.mean()
        else:
            self.mean += self.alpha * activation
        self._compute_membrane_potential(activation)

        return self.membrane

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):
        return self._plot_rfs(self.rfs, *args, **kwargs)
