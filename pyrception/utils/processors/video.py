from typing import Callable
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


class VideoRecorder:

    def __init__(
        self,
        fpath: str | Path,
        height: int,
        width: int,
        function: Callable,
        codec: str | None = None,
        fps: int = 30,
    ):
        """
        Create an animation from a buffer of stored activations.

        Args:
            fpath: Local file path.
            height: Frame height.
            width: Frame width.
            function: Frame factory function.
            codec: The codec to use.
            fps: FPS for the output video.
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

        self.height = height
        self.width = width
        self.canvas = np.zeros((height, width))

        # TODO:
        # Add some logic to check if the value of the 'codec' parameter can be used.
        self.codec = av.Codec(codec or "h264", "w").name
        self.fps = fps
        self.frame_function = function

        self.container = av.open(fpath, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=fps)
        self.stream.height = height
        self.stream.width = width
        self.stream.pix_fmt = "yuv420p"

    @property
    def size(self) -> tuple[int, int]:
        """
        Return the size of the frame.

        Returns:
            The size as a tuple (height, width)
        """
        return (self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, *args):

        # Flush the container and close it.
        packet = self.stream.encode(None)
        self.container.mux(packet)
        self.container.close()

    def update(self):

        frame = ski.exposure.rescale_intensity(
            self.frame_function(), out_range=np.uint8
        )
        if len(frame.shape) == 2:
            frame = ski.color.gray2rgb(frame)

        out_frame = av.VideoFrame.from_ndarray(frame)
        out_packet = self.stream.encode(out_frame)
        self.container.mux(out_packet)


class VideoLoader:

    def __init__(
        self,
        path: str | Path,
        greyscale: bool = True,
        scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int = 30,
        codec: str = "h264",
    ):
        """
        A convenience class for loading video files or directories of images as frames.

        Args:
            path: Save path.
            greyscale: Greyscale toggle
            scale: Scale of the frame.
            height: Frame height.
            width: Frame width.
            fps: The FPS of the video.
            codec: The video codec.

        Raises:
            Raised if both the scale and ether the width or the height are specified.
        """

        self.path = Path(path).resolve().absolute()
        self.greyscale = greyscale
        self.scale = scale
        self.height = height
        self.width = width
        self.fps = fps
        self.codec = codec
        self.frame_count = 0

        if all(p is not None for p in [self.scale, (self.width or self.height)]):
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
        return (self.height, self.width)

    def __iter__(self):
        return self._iterator

    def __repr__(self):
        return f"<{self.__class__.__qualname__} | {self.path.name} | {self.width}x{self.height}@{self.fps} fps | {self.frame_count} frames>"

    def _load(self):
        """
        Load a video file or an image sequence.
        """

        if self.path.is_dir():
            # Load a directory of frames.
            return self._load_frames()

        elif self.path.is_file():
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

        if all(p is None for p in [self.scale, self.width, self.height]):
            return

        if self.scale is not None:
            self.height = int(height * self.scale)
            self.width = int(width * self.scale)

        else:
            # Scale the height proportionally to the width
            if self.width is not None and self.height is None:
                self.height = int(height * self.width / width)

            # Scale the width proportionally to the height
            if self.width is None and self.height is not None:
                self.width = int(width * self.height / height)

    def _load_frames(self):
        """
        Load a directory of frames.
        """

        # Collect the file paths for all images found in the directory.
        image_extensions = {ext.lower() for ext in Image.registered_extensions()}
        fpaths = sorted(
            [
                path
                for path in self.path.glob("*.*")
                if path.suffix.lower() in image_extensions
            ]
        )
        self.frame_count = len(fpaths)

        # Get the original dimensions
        # and compute the scaled ones.
        first_frame = ski.io.imread(fpaths[0])
        shape = first_frame.shape
        height, width = shape[0], shape[1]
        channels = shape[2] if len(shape) > 2 else None
        self._compute_dimensions(height, width)

        options = {}
        effects = []
        if self.greyscale:
            options["as_gray"] = True

        effects.append(partial(ski.io.imread, **options))

        if self.scale is not None:
            scale = [self.scale, self.scale]
            if channels is not None and not self.greyscale:
                scale.append(1)
            effects.append(
                partial(
                    ski.transform.rescale,
                    scale=scale,
                    anti_aliasing=True,
                )
            )

        elif self.width is not None or self.height is not None:
            effects.append(
                partial(ski.transform.resize, output_shape=(self.height, self.width))
            )

        # Update the height and the width
        self.height = self.height or height
        self.width = self.width or width

        def _process(effects: list, frame: Path):
            for effect in effects:
                frame = effect(frame)
            return frame

        return map(partial(_process, effects), fpaths)

    def _load_video(self):
        """
        Load a video clip.
        """

        container = av.open(self.path)
        video = container.streams.video[0]

        self.fps = video.average_rate
        width = video.width
        height = video.height
        self._compute_dimensions(height, width)

        self.codec = video.codec.name
        self.fps = video.base_rate
        self.frame_count = video.frames
        self.duration = float(video.duration * video.time_base)

        options = {}
        if self.greyscale:
            options["format"] = "gray"

        if self.width is not None or self.height is not None:
            options["width"] = self.width
            options["height"] = self.height
            options["interpolation"] = av.video.reformatter.Interpolation.BICUBIC

        # Update the height and the width
        self.height = self.height or height
        self.width = self.width or width

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
            iio.imread(f"{self.path}", index=index, plugin="pyav"), display=True
        )
