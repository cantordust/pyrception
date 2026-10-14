from typing import Callable, Any
import av
from pathlib import Path
import numpy as np
import skimage as ski
from functools import partial
from PIL import Image
import imageio.v3 as iio
from bokeh.plotting import figure

from pyrception import conf
from pyrception import logger
from pyrception import utils
from pyrception.utils import timestamp


class VideoLoader:

    def __init__(
        self,
        vpath: str | Path,
        greyscale: bool = True,
        scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int = 30,
        codec: str = "mpeg4",
    ):
        """
        A convenience class for loading video files or directories of images as frames.

        Args:
            vpath: Path to the video source.
            greyscale: Greyscale toggle
            scale: Scale of the frame.
            height: Frame height.
            width: Frame width.
            fps: The FPS of the video.
            codec: The video codec.

        Raises:
            Raised if both the scale and ether the width or the height are specified.
        """

        self._src_path = Path(vpath).resolve().absolute()
        self._greyscale = greyscale
        self._scale = scale
        self._height = height
        self._width = width
        self._fps = fps
        self._codec = codec
        self._frame_count = 0
        self._duration = 0.0

        if all(p is not None for p in [self._scale, (self._width or self._height)]):
            raise AttributeError(
                f"Please set *either* the new scale *or* the new width and/or height."
            )

        # Internal attributes
        self._containers = {}
        self._iterator = self._load()

    @property
    def size(self) -> tuple[int, ...]:
        """
        Frame size.

        Returns:
            The frame size as a tuple (height, width).
        """
        return (self._height, self._width)

    def __iter__(self):
        return self._iterator

    def __repr__(self):
        return f"<{self.__class__.__qualname__} | {self._src_path.name} | {self._width}x{self._height}@{self._fps} fps | {self._frame_count} frames>"

    def _load(self):
        """
        Load a video file or an image sequence.
        """

        if self._src_path.is_dir():
            # Load a directory of frames.
            return self._load_frames()

        elif self._src_path.is_file():
            # Load an RGB video file.
            return self._load_video()

        else:
            raise TypeError(
                f"Invalid input type. Please use a video or a directory of frames."
            )

    def _compute_dimensions(
        self,
        height: int | None = None,
        width: int | None = None,
    ):
        """
        Compute the new dimensions of the video (if requested).

        Args:
            height: Original height.
            width: Original width.
        """

        if all(p is None for p in [self._scale, self._width, self._height]):
            return

        if self._scale is not None:
            self._height = int(height * self._scale)
            self._width = int(width * self._scale)

        else:
            # Scale the height proportionally to the width
            if self._width is not None and self._height is None:
                self._height = int(height * self._width / width)

            # Scale the width proportionally to the height
            if self._width is None and self._height is not None:
                self._width = int(width * self._height / height)

    def _load_frames(self):
        """
        Load a directory of frames.
        """

        # Collect the file paths for all images found in the directory.
        image_extensions = {ext.lower() for ext in Image.registered_extensions()}
        fpaths = sorted(
            [
                path
                for path in self._src_path.glob("*.*")
                if path.suffix.lower() in image_extensions
            ]
        )
        self._frame_count = len(fpaths)

        # Get the original dimensions
        # and compute the scaled ones.
        first_frame = ski.io.imread(fpaths[0])
        shape = first_frame.shape
        height, width = shape[0], shape[1]
        channels = shape[2] if len(shape) > 2 else None
        self._compute_dimensions(height, width)

        options = {}
        effects = []
        if self._greyscale:
            options["as_gray"] = True

        effects.append(partial(ski.io.imread, **options))

        if self._scale is not None:
            scale = [self._scale, self._scale]
            if channels is not None and not self._greyscale:
                scale.append(1)
            effects.append(
                partial(
                    ski.transform.rescale,
                    scale=scale,
                    anti_aliasing=True,
                )
            )

        elif self._width is not None or self._height is not None:
            effects.append(
                partial(ski.transform.resize, output_shape=(self._height, self._width))
            )

        # Update the height and the width
        self._height = self._height or height
        self._width = self._width or width

        def _process(effects: list, frame: Path):
            for effect in effects:
                frame = effect(frame)
            return frame

        return map(partial(_process, effects), fpaths)

    def _load_video(self):
        """
        Load a video clip.
        """

        container = av.open(self._src_path)
        video = container.streams.video[0]

        self._fps = video.average_rate
        width = video.width
        height = video.height
        self._compute_dimensions(height, width)

        self._codec = video.codec.name
        self._fps = video.base_rate
        self._frame_count = video.frames
        self._duration = float(video.duration * video.time_base)

        options = {}
        if self._greyscale:
            options["format"] = "gray"

        if self._width is not None or self._height is not None:
            options["width"] = self._width
            options["height"] = self._height
            options["interpolation"] = av.video.reformatter.Interpolation.BICUBIC

        # Update the height and the width
        self._height = self._height or height
        self._width = self._width or width

        def _process(options: dict, frame: av.frame.Frame):
            return frame.to_rgb().to_ndarray(channel_last=True, **options)

        return map(partial(_process, options), container.decode(video=0))

    def preview(self, index: int = 0) -> figure:
        """
        Preview the first frame of the input.

        Returns:
            A frame as a NumPy array.
        """
        return utils.plot.image(
            iio.imread(f"{self._src_path}", index=index, plugin="pyav"), display=True
        )


class VideoRecorder:

    def __init__(
        self,
        fpath: str | Path,
        height: int,
        width: int,
        producer: Callable,
        producer_opts: dict[str, Any] | Callable | None = None,
        transformations: list[Callable] | None = None,
        codec: str | None = None,
        fps: int | None = None,
        vl: VideoLoader | None = None,
        name: str | None = None,
    ):
        """
        Create an animation from a buffer of stored activations.

        Args:
            fpath: File path for the final video output.
            height: Frame height.
            width: Frame width.
            producer: Frame producer (a callable taking a 2D NumPy array as an argument).
            producer_opts: Optional dictionary providing keyword arguments for the producer.
            transformations: A list of transformations to be applied to the canvas.
            codec: The codec to use.
            fps: FPS for the output video.
            vl: An optional video loader that can be used to synchronise the codec and the FPS.
            name: An optional name for this recorder.
                If `None`, the one from the video loader will be used (which might also be `None`).
        """

        ts = timestamp()

        if fpath is None:
            fpath = conf.paths.local / f"export-{ts}.mp4"

        fpath = Path(fpath)

        if fpath.exists():
            if fpath.is_dir():
                raise ValueError(
                    f"The specified path is a directory. Please provide a valid file path."
                )
        else:
            fpath.parent.mkdir(exist_ok=True, parents=True)

        self._height = height
        self._width = width
        self._canvas = np.zeros((height, width, 3))

        # If a video loader is passed, synchronise the FPS and the codec.
        self._fps = fps or (vl._fps if vl is not None else 30)

        # TODO: Add some logic to check if the value of the 'codec' parameter can be used.
        codec = codec or (vl._codec if vl is not None else "mpeg4")
        self._codec = av.Codec(codec, "w")

        self._producer = producer

        # Producer options
        if producer_opts is None:
            producer_opts = {}

        if isinstance(producer_opts, dict):
            _opts = lambda: producer_opts

        self._opts = _opts

        self._transformations = [] if transformations is None else transformations

        self._container = av.open(fpath, mode="w")
        self._stream = self._container.add_stream(self._codec.name, rate=fps)
        self._stream.height = height
        self._stream.width = width

        self._vl = vl
        self._name = name

    @property
    def producer_opts(self) -> dict:
        return self._opts()

    @property
    def size(self) -> tuple[int, int]:
        """
        Return the size of the frame.

        Returns:
            The size as a tuple (height, width)
        """
        return (self._height, self._width)

    def __enter__(self):
        return self

    def __exit__(self, *args):

        # Flush the container and close it.
        packet = self._stream.encode(None)
        self._container.mux(packet)
        self._container.close()

    def update(self):

        canvas = self._producer(self._canvas, **self.producer_opts)

        for t in self._transformations:
            canvas = t(canvas)

        canvas = ski.exposure.rescale_intensity(canvas, out_range=np.uint8)

        if len(canvas.shape) == 2:
            canvas = ski.color.gray2rgb(canvas)

        frame = av.VideoFrame.from_ndarray(canvas)
        packet = self._stream.encode(frame)
        self._container.mux(packet)
