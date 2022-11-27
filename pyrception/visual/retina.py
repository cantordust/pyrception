from typing import Any
from typing import Set
from typing import Dict
from typing import Union
from typing import Optional

# --------------------------------------
from loguru import logger

# --------------------------------------
from datetime import datetime as dt

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt

# --------------------------------------
import imageio.v3 as iio

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import View
from pyrception.visual.proto import ProtoLayer
from pyrception.visual.receptor import ReceptorLayer
from pyrception.visual.bipolar import BipolarLayer
from pyrception.visual.ganglion import GanglionLayer


class Retina(ProtoLayer):

    """
    A retinal layer aims to emulate the operation of the retina
    with separate ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        source: Union[str, int],
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
        receptor_args: Optional[Dict[str, Any]] = None,
        bipolar_args: Optional[Dict[str, Any]] = None,
        ganglion_args: Optional[Dict[str, Any]] = None,
        layer_name: str = "Retina",
    ):
        logger.info(f"==[ {layer_name:<8s} ] Initialising layer...")

        # Sanity checks
        if scaled_height is not None and scaled_width is not None:
            raise ValueError("Please provide the new height *or* width, but not both.")

        # * Source parameters * #
        self.src_path = source
        self.source = None
        self.stream = None
        self.reader = None

        # * Current frame * #
        self.frame = None
        self.generator = None

        # * Output channels * #
        self.frame_path = None
        self.video_path = None
        self.video_sink = None

        # * Flag indicating whether the source is still being processed * #
        self.processing = True

        if isinstance(source, str):

            self.src_path = Path(source).absolute()

            print(f"==[ self.src_path: {self.src_path}")

            if self.src_path.is_dir():
                print(f"==[ Using a directory of images as a source")
                self.generator = self._iterate_frames()
                self.reader = self._read_frame_file

            elif self.src_path.is_file():
                print(f"==[ Using a video file as a source")

                # Source stream
                self.stream = cv.VideoCapture(source)
                self.reader = self.stream.read

            else:
                raise ValueError(f"Invalid source type: {type(source)}")

        # A function used to extract a source frame
        elif isinstance(source, int):
            self.stream = cv.VideoCapture(source)
            self.reader = self.stream.read

        else:
            raise TypeError(f"Invalid input source type '{type(source)}'.")

        self._get_frame(probe=True)
        original_shape = list(self.frame.shape)

        # * Receptors + horizontal cells * #
        if receptor_args is None:
            receptor_args = {}

        self.receptor = ReceptorLayer(
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
            self.receptor,
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

        logger.info(f"==[ {layer_name:<8s} ] Initialisation complete.")

        # Display some useful info
        logger.info("Press ESC to quit.")

    def __del__(self):

        self.processing = False

        if self.stream is not None:
            self.stream.release()

        if self.video_sink is not None:
            self.video_sink.release()

        cv.destroyAllWindows()

    def _iterate_frames(self):

        for file in sorted(self.src_path.iterdir()):
            # print(f"==[ file.name: {file.name}")
            if file.is_file() and file.suffix in (".png", ".jpg", ".jpeg"):
                yield iio.imread(file)

    def _read_frame_file(self):

        try:
            frame = next(self.generator)
        except StopIteration:
            self.processing = False
            frame = None

        finally:
            return (self.processing, frame)

    def _get_frame(
        self,
        probe: Optional[bool] = False,
    ):
        """
        Extract the next frame.
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
                (self.receptor.dim.W, self.receptor.dim.H),
                interpolation=cv.INTER_AREA,
            )

        self.frame = pt.from_numpy(self.frame).float()

        return self.frame

    def compose(
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
            pt.zeros_like(views[View.GanglionOffOn]),
        ]

        receptor = np.vstack(
            [ProtoLayer.scale(view).numpy() for view in receptor_views]
        ).astype(np.uint8)

        bipolar = np.vstack(
            [ProtoLayer.scale(view).numpy() for view in bipolar_views]
        ).astype(np.uint8)

        ganglion = np.vstack(
            [ProtoLayer.scale(view).numpy() for view in ganglion_views]
        ).astype(np.uint8)

        image = np.hstack((receptor, bipolar, ganglion))
        cimage = cv.merge((image, image, image))

        # Optical flow
        events = pt.zeros(*views[View.OnOffEvents].shape, 3)
        events[:, :, 0] = pt.relu(views[View.GanglionOnOff])
        events[:, :, 2] = pt.relu(views[View.GanglionOffOn])

        events = 255 * (events - events.min()) / (events.max() - events.min() + 1e-8)
        views[View.OnOffEvents] = events

        cimage[
            2 * cimage.shape[0] // 3 :, 2 * cimage.shape[1] // 3 :, :
        ] = events.numpy().astype(np.uint8)

        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 0.45
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2

        w_offset = 0
        h_offset = 10

        cv.putText(
            cimage,
            # "Raw signal (receptor)",
            "(a)",
            (w_offset, h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "Local mean (horizontal)",
            "(b)",
            (w_offset, self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "Spatial filter",
            "(c)",
            (w_offset, 2 * self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "Temporal filter",
            "(d)",
            (self.receptor.dim.W + w_offset, h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "ON-type bipolar",
            "(e)",
            (self.receptor.dim.W + w_offset, self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "OFF-type bipolar",
            "(f)",
            (self.receptor.dim.W + w_offset, 2 * self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "ON-centre/OFF-surround ganglion",
            "(g)",
            (2 * self.receptor.dim.W + w_offset, h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "OFF-centre/ON-surround ganglion",
            "(h)",
            (2 * self.receptor.dim.W + w_offset, self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv.putText(
            cimage,
            # "ON/OFF (blue) and OFF/ON (red) events",
            "(i)",
            (2 * self.receptor.dim.W + w_offset, 2 * self.receptor.dim.H + h_offset),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        return cimage

    def run(
        self,
        saccades: bool = False,
        show: bool = True,
        output_path: Optional[Path] = None,
        save_frames: Optional[Union[Set[int], bool]] = None,
        save_views: Optional[Dict[View, Optional[Path]]] = None,
        save_video: bool = False,
    ):

        if not saccades:
            saccades = None

        offset = (0.0, 0.0) if saccades else None

        n_frame = 0

        frame_paths = None
        video_path = None

        # * Frame views to save * #

        if save_views is None:
            save_views = {view for view in View}

        # * Paths for saving frames and video * #

        if output_path is not None:

            output_path = Path(output_path).absolute()
            if Path.is_file(output_path):
                output_path = output_path.parent

        else:
            dir = Path(__file__).absolute().parent
            output_path = dir / "output"

        if save_video or (save_frames is not None):

            # Current date and time
            now = dt.now().strftime("%d-%b-%Y_%H:%M:%S")

            # Root output path and paths for video and frames
            root = output_path.absolute() / now

            if save_frames not in (None, False):

                # Save each frame of each view as a separate file
                frame_paths = {}

                for view in save_views:

                    view_path = root / f"{view.name}/frames"
                    Path.mkdir(
                        view_path,
                        parents=True,
                        exist_ok=True,
                    )

                    frame_paths[view] = view_path

            if save_video:

                # Save the output as a video file
                video_path = root / "video.mp4"

                Path.mkdir(
                    root,
                    parents=True,
                    exist_ok=True,
                )

                print(f"==[ video_path: {video_path}")

                format = "mp4v"
                fourcc = cv.VideoWriter_fourcc(*format)

                self.video_sink = cv.VideoWriter(
                    str(video_path),
                    fourcc,
                    30.0,
                    (
                        self.receptor.dim.W * 3,
                        self.receptor.dim.H * 3,
                    ),
                )

        # * Frame set to save * #

        save_all_frames = False

        if save_frames in (True, False, None):

            if save_frames == True:
                save_all_frames = True

            save_frames = set()

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

            # Process input via receptors + horizontal cells
            self.receptor.process(
                frame,
                offset,
                views,
                n_frame,
                save_frames,
                save_views,
                frame_paths,
            )

            # Pass through bipolar layer
            self.bipolar.process(
                views,
                n_frame,
                save_frames,
                save_views,
                frame_paths,
            )

            # Pass through amacrine and ganglion layers
            self.ganglion.process(
                views,
                n_frame,
                save_frames,
                save_views,
                frame_paths,
            )

            # Composite image
            cimage = self.compose(views)

            if show:

                # Show all images
                cv.imshow(f"Result", cimage)

                # Press ESC to quit
                self.processing &= cv.waitKey(10) != 27

            if (n_frame in save_frames) or save_all_frames:
                self._save_views(
                    # {View.Composite: pt.from_numpy(cimage)},
                    views,
                    n_frame,
                    save_views,
                    frame_paths,
                )

            if self.video_sink is not None:
                self.video_sink.write(cimage)
