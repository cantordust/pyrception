import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
from pyrception import conf
from pyrception.conf import logger
from pyrception.visual.util.types import KernelFilter
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.layers.receptor import ReceptorLayer
from pyrception.visual.layers.horizontal import HorizontalLayer
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.util.types import RFArrangement

class BipolarLayer(BaseLayer):
    """
    A layer of bipolar cells.
    This layer processes the signal form the receptor layer
    modulated by the horizontal layer.
    """

    def __init__(
        self,
        shape: tp.Tuple[int, ...],
        receptor: ReceptorLayer,
        horizontal: HorizontalLayer,
        sectors: int = 64,
        name: str = "Bipolar",
        forgetting_range: tp.Tuple[float, float] = (0.05, 0.95),
        rf_params: tp.Dict[str, tp.Any] = None,
    ):

        # Initialise the base
        super().__init__(shape, name)
        self.receptor = receptor
        self.horizontal = horizontal

        # Initialise the receptive fields.
        if rf_params is None:
            rf_params = {}
        rf_params.setdefault("name", f"{name} | Bipolar RFs")
        self.rfs = ReceptiveFields(
            self.shape,
            receptor.rfs.cell_coordinates,
            sectors,
            **rf_params,
        )
        self.rfs.make_rfs()

        # Temporal exponential running mean
        self.mean = np.zeros((self.rfs.neuron_count,))

        self.forgetting_range = forgetting_range

        # 'Forgetting rate' for the temporal mean
        self.alpha = self._compute_forgetting_rate()

        # Activations for the ON and OFF pathways
        self.on = np.zeros((self.rfs.neuron_count,))
        self.off = np.zeros_like(self.on)

        self.info("Initialised.")

    def _compute_forgetting_rate(self):
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

    def forward(self) -> tp.Tuple[np.ndarray, ...]:
        """
        The bipolar layer splits the input into ON and OFF pathways.

        Returns:
            tp.Tuple[np.ndarray, ...]:
                A tuple containing:
                    1. The activation of ON bipolar cells.
                    2. The activation of OFF bipolar cells.
                    3. The scaled signal (raw input signal sans feedback from the horizontal cells).
                    4. The raw (postsynaptic) activation of the bipolar cells.
        """

        # Subtract the raw photoreceptor signal
        # from the horizontal feedback signal
        scaled_signal = self.receptor.activation - self.horizontal.feedback

        # Compute the nominal activation for each bipolar cell
        activation = self.convolve(self.rfs.forward_synapses, scaled_signal)

        # Compute the on and off activations
        diff = activation - self.mean
        self.on = np.log1p(np.where(diff > 0, diff, 0))
        self.off = np.log1p(np.where(diff < 0, -diff, 0))

        # Update the running mean (low-pass filter)
        self.mean += self.alpha * diff

        return (self.on, self.off, scaled_signal, activation)

    def plot_rfs(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("title", "Bipolar layer receptive fields")
        return self._plot_rfs(self.rfs, *args, **kwargs)
