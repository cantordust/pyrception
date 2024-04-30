from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelType
from pyrception.visual.layers.horizontal import HorizontalLayer
from pyrception.visual.layers.proto import ProtoLayer


class BipolarLayer(ProtoLayer):
    """
    A layer of bipolar cells.
    This layer processes the signal form the receptor layer
    modulated by the horizontal layer.
    """

    def __init__(
        self,
        alpha: Union[pt.Tensor, float, int],
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Bipolar")
        super().__init__(*args, **kwargs)

        # ON and OFF pathways
        self.on = pt.zeros((self.neuron_count,))
        self.off = pt.zeros_like(self.on)

        if not isinstance(alpha, (pt.Tensor, float, int)):
            raise TypeError(
                f"The alpha parameter should be a numeric or tensor type (got {type(alpha)})."
            )

        if isinstance(alpha, (float, int)):

            # Numeric
            if not (0.0 < alpha < 1.0):
                raise ValueError(
                    f"The value of alpha should be between 0 and 1 (got {alpha})."
                )
            alpha = pt.full((self.neuron_count,), float(alpha), device=conf.device)

        else:
            # Tensor
            if not (pt.equal(alpha.shape, self.on.shape)):
                raise TypeError(
                    f"The alpha tensor must have the same shape as the ON tensor ({self.on.shape}; got {self.alpha.shape})"
                )

            alpha = alpha.to(conf.device)

        # Exponential running mean (temporal mean)
        # ==================================================
        self.on_mean = pt.zeros((self.neuron_count,), device=conf.device)
        self.off_mean = pt.zeros((self.neuron_count,), device=conf.device)

        # 'Forgetting rate' for the temporal mean
        self.alpha = alpha

        self.info("Initialised.")

    def __call__(
        self,
        raw: pt.Tensor,
        horizontal: pt.Tensor,
    ) -> pt.Tensor:
        """
        Split the input into ON and OFF pathways.
        """

        # Subtract the raw photoreceptor signal
        # from the running (temporal) mean
        scaled = raw - horizontal

        activation = self.convolve(scaled)

        on = pt.relu(activation)
        off = pt.relu(-activation)

        # Update the running mean
        self.on_mean += self.alpha * (on - self.on_mean)
        self.off_mean += self.alpha * (off - self.off_mean)

        return (on, off)
