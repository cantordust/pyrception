# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception.visual.util.types import View
from pyrception.visual.proto import ProtoLayer
from pyrception.visual.receptor import ReceptorLayer
from pyrception.visual.bipolar import BipolarLayer
from pyrception.visual.ganglion import GanglionLayer


class Retina:

    """
    A retinal layer aims to emulate the operation of the retina
    with separate ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        source: Union[np.ndarray, cv.VideoCapture],
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
        receptor_args: Optional[Dict[str, Any]] = None,
        bipolar_args: Optional[Dict[str, Any]] = None,
        ganglion_args: Optional[Dict[str, Any]] = None,
    ):

        # Flag indicating whether the source is still being processed
        self.processing = True

        # Current frame
        self.frame = None

        # Store the source
        self.source = source

        # Compute source and output dimensions
        if scaled_height is not None and scaled_width is not None:
            raise ValueError("Please provide the new height *or* width, but not both.")

        # A function used to extract a source frame
        if not isinstance(source, (np.ndarray, cv.VideoCapture)):
            raise TypeError(f"Invalid input source type '{type(source)}'.")

        elif isinstance(source, cv.VideoCapture):
            self.frame_op = lambda processing, src: src.read()

        if isinstance(source, np.ndarray):
            self.frame = source
            self.frame_op = lambda processing, src: (processing, self.frame)

        self._get_frame(probe=True)
        original_shape = list(self.frame.shape)

        # * Receptors + horizontal cells * #
        if receptor_args is None:
            receptor_args = {}

        self.receptors = ReceptorLayer(
            original_shape,
            scaled_height,
            scaled_width,
            saccades,
            **receptor_args,
        )

        # print(f"==[ receptors: {self.receptors.kmask.shape}")

        # * Bipolar cells * #
        if bipolar_args is None:
            bipolar_args = {}

        self.bipolar = BipolarLayer(
            self.receptors,
            saccades,
            **bipolar_args,
        )

        # * Amacrine + ganglion cells * #

        if ganglion_args is None:
            ganglion_args = {}

        self.ganglion = GanglionLayer(
            self.bipolar,
            saccades,
            **ganglion_args,
        )

        # print(f"==[ bipolar: {self.bipolar.kmask.shape}")

        # Display some useful info
        print(f"==[ Press ESC to quit.")

    def __del__(self):

        self.processing = False
        cv.destroyAllWindows()

        if isinstance(self.source, cv.VideoCapture):
            self.source.release()

    def _get_frame(
        self,
        probe: Optional[bool] = False,
    ):
        """
        Extract a frame at a certain offset from the center.
        """

        # Get the next frame by applying self.frame_op to the source
        self.processing, self.frame = self.frame_op(self.processing, self.source)

        self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2GRAY)

        if probe:
            self.processing = True
            return self.frame

        if self.receptors.dim.resize:
            self.frame = cv.resize(
                self.frame,
                (self.receptors.dim.W, self.receptors.dim.H),
                interpolation=cv.INTER_AREA,
            )

        self.frame = pt.from_numpy(self.frame).float()

        return self.frame

    def show(
        self,
        views: Set[pt.Tensor],
    ):

        if len(views) == 0:
            return

        receptor_views = [
            views[View.Original],
            views[View.ReceptorMean],
            views[View.ReceptorAdapted],
        ]

        bipolar_views = [
            views[View.BipolarMean],
            views[View.BipolarOn],
            views[View.BipolarOff],
        ]

        ganglion_views = [
            views[View.GanglionOnOff],
            views[View.GanglionOffOn],
            pt.zeros_like(views[View.GanglionOnOff]),
        ]

        receptor = np.hstack(
            [ProtoLayer.scale(view).numpy() for view in receptor_views]
        ).astype(np.uint8)

        bipolar = np.hstack(
            [ProtoLayer.scale(view).numpy() for view in bipolar_views]
        ).astype(np.uint8)

        ganglion = np.hstack(
            [ProtoLayer.scale(view).numpy() for view in ganglion_views]
        ).astype(np.uint8)

        image = np.vstack((receptor, bipolar, ganglion))

        # Show all images
        cv.imshow(f"Result", image)

        # Press ESC to quit
        self.processing &= cv.waitKey(10) != 27

    def run(
        self,
        show: bool = True,
        saccades: bool = False,
    ):

        if not saccades:
            saccades = None

        offset = (0.0, 0.0) if saccades else None

        while self.processing:

            if saccades and np.random.random() <= 0.05:
                height_offset = (np.random.random(1) - 0.5) / 4
                width_offset = (np.random.random(1) - 0.5) / 4
                offset = (
                    height_offset,
                    width_offset,
                )

            frame = self._get_frame()
            views = self.receptors.process(frame, offset)
            views.update(self.bipolar.process(views[View.ReceptorAdapted]))
            views.update(
                self.ganglion.process(views[View.BipolarOn], views[View.BipolarOff])
            )

            if show:
                self.show(views)
