import typing as tp

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtCore import Signal
from PySide6.QtCore import QEvent
from PySide6.QtCore import QObject

from PySide6.QtGui import QKeyEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtGui import QEnterEvent
from PySide6.QtGui import QTransform

from PySide6.QtWidgets import QAbstractSpinBox
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QSizePolicy

# --------------------------------------
import numpy as np

# --------------------------------------
from pathlib import Path

# --------------------------------------
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from pyqtgraph import SignalProxy
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree import Parameter

# --------------------------------------
from pyrception.conf import logger
from pyrception.visual import InputType
from pyrception.util.functions import load_image
from pyrception.util.functions import load_video
from pyrception.gui import ImageView


class ResourceLoader(QWidget):

    def __init__(
        self,
        path: Path,
        itype: InputType,
        callback: tp.Callable,
        grayscale: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.setFixedSize(640, 480)

        self.path = path

        self.grid = QGridLayout(self)
        self.iv = ImageView()
        self.iv.getView().setMouseEnabled(x=False, y=False)
        self.iv.getView().setMenuEnabled(False)
        self.grid.addWidget(self.iv, 0, 0, 1, 10)

        self.frames: np.ndarray = None

        self.orig_h = 0
        self.orig_w = 0

        self.w = 0
        self.h = 0

        self.scale_label = QLabel("Scale:")
        self.height_label = QLabel()
        self.width_label = QLabel()
        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setValue(1.0)
        self.spinbox.setDecimals(2)
        self.spinbox.setRange(0.05, 5)
        self.spinbox.setSingleStep(0.01)

        self.ok = QPushButton("OK")
        self.callback = callback
        self.ok.pressed.connect(self.ok_pressed)

        self.spinbox.valueChanged.connect(self.rescale)

        self.grid.addWidget(self.width_label, 1, 0, 1, 1)
        self.grid.addWidget(self.height_label, 2, 0, 1, 1)
        self.grid.addWidget(self.scale_label, 3, 0, 1, 1)
        self.grid.addWidget(self.spinbox, 3, 1, 1, 1)
        self.grid.addWidget(self.ok, 4, 0, 1, 1)

        self._load(path, itype, grayscale, scale)

    def update_labels(self):
        self.height_label.setText(f"Height: {self.h}px")
        self.width_label.setText(f"Width: {self.w}px")

    @Slot(float)
    def rescale(self, scale: float):

        self.h = int(self.orig_h * scale)
        self.w = int(self.orig_w * scale)

        self.update_labels()

    @Slot()
    def ok_pressed(self):

        self.callback(self.spinbox.value())
        self.close()

    def _load(
        self,
        path: Path,
        itype: InputType,
        grayscale: bool = True,
        scale: float = 1.0,
    ):

        if itype == InputType.Image:
            logger.info(f"Loading image '{path}'")
            frames = load_image(path, grayscale, scale)

        elif itype == InputType.Video:
            logger.info(f"Loading video '{path}'")
            frames = load_video(path, grayscale, scale, probe=True)

        self.frames = frames
        self.iv.setImage(self.frames[0])

        self.orig_h = self.frames[0].shape[0]
        self.orig_w = self.frames[0].shape[1]
        self.h = self.orig_h
        self.w = self.orig_w

        self.update_labels()
