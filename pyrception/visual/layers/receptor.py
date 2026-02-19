import numpy as np
from pathlib import Path
from pyrception import utils
from pyrception.visual.layers.base import LayerBase
from pyrception.utils.processors import VideoLoader
from pyrception.utils.processors import VideoRecorder

class ReceptorLayer(LayerBase):
    """
    A layer of receptors.

    This layer applies the first processing steps to the raw input,
    including saccades and RGB to grayscale conversion.
    """

    def __init__(
        self,
        size: tuple[int, ...] | np.ndarray,
        scale: float = 1.0,
        greyscale: bool = True,
        log_response: bool = True,
        substrate: np.ndarray | None = None,
        activations: np.ndarray | None = None,
        *args,
        **kwargs,
    ):

        super().__init__(size, *args, **kwargs)

        size = utils.arg2np(size, ext=3)
        self.size = size[:2]
        self.scale = scale
        self.greyscale = greyscale
        self.log_response = log_response
        self.substrate = substrate
        self.activations = activations
        self.frame = np.zeros(self.size)

        # Internal attributes used for constructing receptive fields.
        # ==================================================
        self.height = self.size[0].item()
        self.width = self.size[1].item()
        self.depth = 1 if len(size.shape) < 3 else size[2].item()
        self.substrate = substrate or utils.make_substrate(self.size[0], self.size[1])
        self.activations = None

        self.logger.info("Initialised.")

    def forward(
        self,
        signal: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Read the input and apply a certain offset if saccades are enabled.

        Args:
            signal: The raw input signal.
            dt: In the case of temporal integration,
                indicates the time since the last input.

        Returns:
            The frame with optional padding (for saccades).
        """

        # Log response
        if self.log_response:
            signal = np.log1p(signal)

        self.frame = signal

        # The activations are just the flattened raw input.
        self.activations = signal.flatten()

        for recorder in self._recorders.values():
            recorder.update()

        return self.activations

    def record_raw(
        self,
        fpath: str | Path | None = None,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        '''
        Add a recorder to the layer.

        Args:
            fpath: Optional path to save the recording to.
            vl: Optional video loader.

        Returns:
            A VideoRecorder instance.
        '''

        return self.add_recorder(
            "raw",
            fpath,
            lambda: self.frame,
            self.size,
            vl=vl,
        )
