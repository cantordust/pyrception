import typing as tp

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import imageio.v3 as iio

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception import conf
from pyrception.logging import Logging
from pyrception.visual.util.types import Dim
from pyrception.visual.util.types import Dims
from pyrception.visual.layers.receptor import ReceptorLayer
from pyrception.visual.layers.horizontal import HorizontalLayer
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.layers.amacrine import AmacrineLayer
from pyrception.visual.layers.ganglion import GanglionLayer


class Retina(Logging):
    """
    A retinal layer aims to emulate the full processing pipeline
    of the mammalian retina, from receptors to ganglion cells.
    """

    def __init__(
        self,
        source: tp.Union[str, int],
        shape: tp.Tuple[int, ...],
        saccades: bool = False,
        name: str = "Retina",
        *args,
        **kwargs,
    ):
        """
        Retina initialisation.

        Args:
            source (tp.Union[str, int]):
                Input source.

            shape (tp.Tuple[int, ...]):
                Dimensions of the visual field.

            saccades (bool, optional):
                Toggle for saccadic movements. Defaults to False.

            name (str, optional):
                Layer name. Defaults to "Retina".

        Raises:
            ValueError:
                Raised if the provided size is invalid.

            TypeError:
                Raised if the provided source is invalid.
        """

        # Initialise the base
        super().__init__(name, *args, **kwargs)

        self.info("Initialising...")

        # Source parameters
        # ==================================================
        self.source = source
        self.shape = shape
        self.saccades = saccades
        self.src_path = None
        self.stream = None
        self.reader = None
        #: Flag indicating whether the source is still being processed
        self.processing = True
        self.frame_path = None
        self.video_path = None
        self.frame = None
        self.generator = None
        self._setup_source()

        # Receptor layer
        # ==================================================
        receptor_args = kwargs.get("receptor_layer", {})
        self.receptor = ReceptorLayer(shape, **receptor_args)

        # Horizontal layer
        # ==================================================
        horizontal_args = kwargs.get("horizontal_layer", {})
        self.horizontal = HorizontalLayer(shape, self.receptor, **horizontal_args)

        # Bipolar layer
        # ==================================================
        bipolar_args = kwargs.get("bipolar_layer", {})
        self.bipolar = BipolarLayer(
            shape, self.receptor, self.horizontal, **bipolar_args
        )

        # Amacrine layer
        # ==================================================
        amacrine_args = kwargs.get("amacrine_layer", {})
        self.amacrine = AmacrineLayer(shape, self.bipolar, **amacrine_args)

        # Ganglion layer
        # ==================================================
        ganglion_args = kwargs.get("ganglion_layer", {})
        self.ganglion = GanglionLayer(
            shape, self.bipolar, self.amacrine, **ganglion_args
        )

        self.info("Initialised.")

        # TODO: Handle cases where we don't use OpenCV
        if source == 0:
            self.info("Press ESC to quit.")

    def __del__(self):
        self.processing = False

        cv.destroyAllWindows()

    def _setup_source(self):

        if isinstance(self.source, str):
            self.src_path = Path(self.source).absolute()

            self.debug(f"Source path: {self.src_path}")

            if self.src_path.is_dir():
                self.debug("Using a directory of images as a source")
                self.generator = self._iterate_frames()
                self.reader = self._read_frame_file

            elif self.src_path.is_file():
                self.debug(f"Using a video file as a source")

                # Source stream
                self.stream = cv.VideoCapture(self.source)
                self.reader = self.stream.read

            else:
                raise TypeError(f"Invalid source type: {type(self.source)}")

        elif isinstance(self.source, int):
            self.stream = cv.VideoCapture(self.source)
            self.reader = self.stream.read

        else:
            raise TypeError(f"Invalid source type '{type(self.source)}'.")

    def _iterate_frames(self):
        """
        Iterate over a collection of image files.

        Yields:
            np.ndarray:
                Frame as a NumPy array.
        """

        for file in sorted(self.src_path.iterdir()):
            if file.is_file() and file.suffix in (".png", ".jpg", ".jpeg"):
                yield iio.imread(file)

    def _read_frame_file(self) -> tp.Tuple[bool, np.ndarray]:
        """
        Read a frame from a video file.

        Returns:
            tp.Tuple[bool, np.ndarray]:
                A tuple containing:
                    1. The processing indicator (if the file is still being read from)
                    2. The frame as a NumPy array
        """

        try:
            frame = next(self.generator)
        except StopIteration:
            self.processing = False
            frame = None

        finally:
            return (self.processing, frame)

    def _get_frame(
        self,
        probe: tp.Optional[bool] = False,
    ):
        """
        _summary_

        Args:
            probe (tp.Optional[bool], optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # Get the current frame
        self.processing, self.frame = self.reader()

        if self.frame is None:
            return

        self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2GRAY)

        if probe:
            self.processing = True
            if self.generator is not None:
                # Reset the generator
                self.generator = self._iterate_frames()
            return self.frame

        if self.receptor.dim.resize:
            self.frame = cv.resize(
                self.frame,
                (
                    self.receptor.dims.original.width,
                    self.receptor.dims.original.height,
                ),
                interpolation=cv.INTER_AREA,
            )

        self.frame = self.frame.astype(conf.dtype)

        return self.frame

    def run(self):
        """
        Run the processing pipeline.
        """
        if not saccades:
            saccades = None

        offset = (0.0, 0.0) if saccades else None

        n_frame = 0

        frame_paths = None
        video_path = None

        # * Frame views to save * #
        while True:
            if saccades and np.random.random() <= 0.05:
                height_offset = (np.random.random(1) - 0.5) / 4
                width_offset = (np.random.random(1) - 0.5) / 4
                offset = (
                    height_offset,
                    width_offset,
                )

            frame = self._get_frame()
            n_frame += 1

            if not self.processing or frame is None:
                break

            views = {}
