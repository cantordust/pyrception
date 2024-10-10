# --------------------------------------
import typing as tp

# --------------------------------------
import multiprocessing as mp

# --------------------------------------
import sys

# --------------------------------------
import numpy as np

# --------------------------------------
from pathlib import Path

# --------------------------------------
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QTabWidget

# --------------------------------------
from pyrception.conf import logger
from pyrception.gui.display.tab import Tab
from pyrception.visual import InputType


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Pyrception")

        # Tab widget
        # ==================================================
        self.tabs = QTabWidget()
        self.setup_tabs()
        self.setCentralWidget(self.tabs)

        # Menu
        # ==================================================
        self.menu = self.menuBar()
        self.main_menu = self.menu.addMenu("File")
        self.setup_menu()

        # Window dimensions
        # ==================================================
        geometry = self.screen().availableGeometry()

        # Slots & signals
        # ==================================================

    def setup_tabs(self):
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._remove_tab)

    def setup_menu(self):
        # Main menu
        # ==================================================

        # Open image
        act_open_image = QAction("Open image", self)
        act_open_image.triggered.connect(self._open_image)
        self.main_menu.addAction(act_open_image)

        # Open video
        act_open_movie = QAction("Open video", self)
        act_open_movie.triggered.connect(self._open_video)
        self.main_menu.addAction(act_open_movie)

        # Open events
        act_open_events = QAction("Open events", self)
        act_open_events.triggered.connect(self._open_events)
        self.main_menu.addAction(act_open_events)

        # Exit
        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence.Quit)
        act_exit.triggered.connect(self.close)
        self.main_menu.addAction(act_exit)

    def _open_image(
        self,
        image_path: Path = None,
    ):

        if not image_path:
            image_filter = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            image_path = QFileDialog.getOpenFileName(
                self,
                "Open file...",
                filter=image_filter,
                selectedFilter=image_filter,
            )[0]

        if image_path != "":
            self._add_tab(Path(image_path), InputType.Image)

    def _open_video(
        self,
        video_path: Path = None,
    ):
        if not video_path:

            video_filter = "Video files (*.avi *.mp4 *.mkv)"
            video_path = QFileDialog.getOpenFileName(
                self,
                "Open video...",
                filter=video_filter,
                selectedFilter=video_filter,
            )[0]

        if video_path != "":
            self._add_tab(Path(video_path), InputType.Video)

    def _open_events(
        self,
        event_path: Path = None,
    ):
        if not event_path:

            event_filter = "Event files (*.raw *.aedat4)"
            event_path = QFileDialog.getOpenFileName(
                self,
                "Open events...",
                filter=event_filter,
                selectedFilter=event_filter,
            )[0]

        if event_path != "":
            self._add_tab(Path(event_path), InputType.Events)

    @Slot(int)
    def _remove_tab(
        self,
        pos: int,
    ):
        if pos < self.tabs.count():
            self.tabs.removeTab(pos)

    def _add_tab(
        self,
        path: Path,
        format: InputType,
    ):
        tab = Tab(path, format)
        idx = self.tabs.addTab(tab, path.name)
        self.tabs.setCurrentIndex(idx)


def run():

    # Set the multiprocessing context
    if "forkserver" in mp.get_all_start_methods():
        mp.set_start_method("forkserver")
    else:
        mp.set_start_method("spawn")

    # The main feature
    app = QApplication([])
    mw = MainWindow()

    img_file = Path("./docs/source/notebooks/resources/cat.jpg").resolve().absolute()
    mw._open_image(img_file)

    mw.showMaximized()
    mw.show()

    sys.exit(app.exec())
