# --------------------------------------
import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import pyqtgraph as pg


class ImageView(pg.ImageView):
    def __init__(
        self,
        frames: np.ndarray = None,
        timeline: bool = False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        if frames is not None:
            self.setImage(frames)

        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        if not timeline:
            self.ui.roiPlot.hide()
            self.axes["t"] = None

        self.view.invertX(False)
        self.view.invertY(True)
        self.ui.splitter.setHandleWidth(2)
