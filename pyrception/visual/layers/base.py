import numpy as np
from typing import Callable
from pyrception import logger
from pyrception.utils.mixins import LoggingMixin
from pathlib import Path
from pyrception.utils.processors import VideoRecorder
from pyrception.utils.processors import VideoLoader
import tempfile


class LayerBase(LoggingMixin):
    """
    Simple base layer implementing methods used by all retinal layers.
    """

    def __init__(
        self,
        size: np.ndarray | None = None,
        source: str | None = None,
        notifier: Callable | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(source, notifier, *args, **kwargs)

        self._size = size
        self._canvas = np.zeros(self._size)
        self._recorders: dict[str, VideoRecorder] = {}

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def convolve(
        self,
        rf: np.ndarray,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Convolve the unrolled input vector with the current layer's receptive field.

        Args:
            rf: Receptive field.
            signal: The input signal (unrolled).

        Returns:
            The convolved signal (unrolled).
        """

        # TODO
        # Alternatives:
        #   - CuPy
        #   - PyTorch
        #   - Sparse (the library)
        #   - Jax(?)
        # ==================================================
        return rf @ signal

    def update_state(self, dt: float, *args, **kwargs):
        """
        Updates the internal state based on temporal dynamics.
        Should be be implemented in derived classes.

        Args:
            dt:
                The time interval since the last input.
        """
        pass

    def forward(self, *args, **kwargs):
        """
        Generic method for processing input.
        Should be overridden in derived classes.
        """
        pass

    def add_recorder(
        self,
        name: str,
        fpath: str | Path,
        function: Callable,
        size: tuple[int, int] | np.ndarray | None = None,
        codec: str | None = None,
        fps: int | None = None,
        vl: VideoLoader | None = None,
    ) -> VideoRecorder:
        """
        Add a video recorder.

        Args:
            name: Name of the recorder.
            fpath: File to save the recorded video output to.
            function: A function producing frames.
            size: Size (height x width) of the frame.
            codec: The codec to use for the VideoRecorder.
            fps: The FPS to use for the VideoRecorder.
            vl: An optional VideoLoader instance.

        Returns:
            A VideoRecorder instance.
        """

        if size is None:
            size = self._size

        options = {}

        # If the video loader is not None,
        # synchronise the FPS and the codec.
        fps = fps or (vl.fps if vl is not None else 30)
        codec = codec or (vl.codec if vl is not None else "h264")
        options["fps"] = fps
        options["codec"] = codec

        recorder = VideoRecorder(fpath, size[0], size[1], function, **options)
        self._recorders[name] = recorder
        return recorder
