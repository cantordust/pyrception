import typing as tp

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtCore import QThread
from PySide6.QtCore import Signal

from PySide6.QtGui import QAction
from PySide6.QtGui import QIcon

from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QToolBar
from PySide6.QtWidgets import QDockWidget
from PySide6.QtWidgets import QProgressBar

# --------------------------------------
from pprint import pp

# --------------------------------------
from pathlib import Path

# --------------------------------------
from pyqtgraph.parametertree import Parameter

# --------------------------------------
from pyrception.conf import logger
from pyrception.visual import InputType
from pyrception.util.functions import load_image
from pyrception.util.functions import load_video
from pyrception.gui.display.canvas import Canvas
from pyrception.gui.display.dock import Dock
from pyrception.gui.display.splitview import SplitView
from pyrception.gui.param import factory as pf


class Tab(QMainWindow):
    load_stack = Signal()

    def __init__(
        self,
        path: Path,
        itype: InputType,
        grayscale: bool = True,
        scale: float = None,
    ):
        super().__init__()

        self.path = path
        self.itype = itype

        # The display canvas
        # ==================================================
        self.canvas: Canvas = None

        # Load the source
        # ==================================================
        self._load(path, itype, grayscale, scale)

        # Splitview widget
        # ==================================================
        self.splitview: SplitView = None

        # Dock
        # ==================================================
        self.dock = Dock(features=QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.setup_dock()
        self.dock.hide()

        # Toolbar
        # ==================================================
        self.toolbar = QToolBar(floatable=False, movable=False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.setup_toolbar()

        # Thread for preventing the GUI from blocking
        # ==================================================
        # self.worker = QThread()
        # self.worker.start()

        # The layer stack
        # ==================================================
        # self.stack = Stack(paths, conf.show_inactive_plots)
        # self.load_stack.connect(self.stack.process)
        # self.stack.set_canvas.connect(self.set_canvas)
        # self.stack.abort.connect(self._abort)
        # self.stack.moveToThread(self.worker)

        # Status bar
        # ==================================================
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setGeometry(30, 40, 200, 25)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.setup_statusbar()

        # Slots & signals
        # ==================================================
        # self.stack.update_progress.connect(self._update_progress_bar)
        # self.dock.sig_show_inactive_plots.connect(self.splitview._show_inactive_plots)
        # self.dock.sig_set_active_plot_colour.connect(
        #     self.splitview._set_active_plot_colour
        # )
        # self.dock.sig_set_inactive_plot_colour.connect(
        #     self.splitview._set_inactive_plot_colour
        # )
        # self.dock.sig_set_stack_contour_colour.connect(
        #     self.canvas._set_stack_contour_colour
        # )
        # self.dock.sig_set_slice_contour_colour.connect(
        #     self.canvas._set_slice_contour_colour
        # )
        # self.dock.sig_show_stack.connect(self.canvas._set_show_stack)

        self.ready = True

    # def __del__(self):
    #     if self.worker.isRunning():
    #         self.worker.quit()

    def _load(
        self,
        path: Path,
        itype: InputType,
        grayscale: bool = True,
        scale: float = None,
    ):

        if itype == InputType.Image:
            logger.info(f"Loading image '{path}'")
            self.canvas = Canvas(load_image(path, grayscale, scale))

        elif itype == InputType.Video:
            logger.info(f"Loading video '{path}'")
            self.canvas = Canvas(load_video(path, grayscale, scale, probe=True))

        self._update_view()

    def _update_view(self):
        if self.canvas is not None:
            self.splitview = SplitView(self.canvas)
            self.splitview.setSizes((2, 1))
            self.setCentralWidget(self.splitview)

    def setup_statusbar(self):
        pass

    def _update_progress_bar(self, path: Path):
        self.status_bar.showMessage(f"Processing {path}")
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def setup_toolbar(self):

        # Dock widget toggle
        act_dock = QAction(
            QIcon.fromTheme(QIcon.ThemeIcon.DocumentProperties),
            "Show docking panel",
            self.toolbar,
        )
        act_dock.triggered.connect(self._toggle_dock)
        self.toolbar.addAction(act_dock)

        # Scale image button
        act_scale_image = QAction(
            QIcon.fromTheme(QIcon.ThemeIcon.ZoomFitBest),
            "Scale image to fit",
            self.toolbar,
        )
        # act_scale_image.triggered.connect(self.canvas._reset_zoom)
        self.toolbar.addAction(act_scale_image)

        # Create layers
        act_create_layers = QAction(
            QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart),
            "Create layers",
            self.toolbar,
        )
        act_create_layers.triggered.connect(
            lambda p: self.canvas.make_layers(self.dock.root)
        )
        self.toolbar.addAction(act_create_layers)

        # # Find centre
        # act_find_centre = QAction(QIcon.fromTheme("tools-media-optical-format"), "Reset centre", self.toolbar)
        # act_find_centre.triggered.connect(lambda: self.canvas._reset_centre())
        # self.toolbar.addAction(act_find_centre)

        # # Radial profile button
        # act_radial_profile = QAction(QIcon.fromTheme("object-rotate-left"), "Plot radial profile", self.toolbar)
        # act_radial_profile.triggered.connect(self.canvas.compute_radial_profile)
        # self.toolbar.addAction(act_radial_profile)

        # # Radial plot button
        # act_plot = QAction(QIcon.fromTheme("list-add"), "Plot radial profile", self.toolbar)
        # act_plot.triggered.connect(lambda: self.splitview.plot(self.stack.current_layer))
        # self.toolbar.addAction(act_plot)

    @Slot()
    def _toggle_dock(self):
        self.dock.setVisible(not self.dock.isVisible())

    # @Slot()
    # def _set_threshold(
    #     self,
    #     update: bool = True,
    # ):
    #     self.dock._trigger_show_inactive_plots()
    #     self.stack._update_threshold(self.dock.show_inactive_plots)

    # def _load(self):
    #     self.status_bar.showMessage(f"Loading stack from {self.paths[0].parent}...")
    #     self.progress_bar.show()
    #     self.load_stack.emit()

    # def _abort(self):
    #     self.worker.quit()

    # @Slot()
    # def set_canvas(self):

    #     self.status_bar.showMessage(f"Setting up canvas...")
    #     self.progress_bar.setValue(0)
    #     self.progress_bar.hide()
    #     self.worker.quit()
    #     self.canvas.set_stack(self.stack, auto_range=True)
    #     self.status_bar.clearMessage()

    @Slot()
    def tree_changed(self):

        self.ready = False

    def setup_dock(self):

        # Input parameters
        # ==================================================
        height_param = pf.make_int(
            "Height",
            self.canvas.frames[0].shape[0],
            limits=[None, None],
            readonly=True,
        )
        width_param = pf.make_int(
            "Width",
            self.canvas.frames[0].shape[1],
            limits=[None, None],
            readonly=True,
        )

        input_parameters = pf.make_group("Input", [height_param, width_param])

        self.dock.add_parameter(input_parameters)

        # Receptor parameters
        # ==================================================
        # Scale
        receptor_scale_param = pf.make_float(
            "Scale",
            1.0,
            limits=[0.05, 5],
            step=0.01,
        )

        @Slot(Parameter)
        def _scale_changed(param: Parameter):
            height_param.setValue(
                int(self.canvas.frames[0].shape[0] * receptor_scale_param.value())
            )
            width_param.setValue(
                int(self.canvas.frames[0].shape[1] * receptor_scale_param.value())
            )

        receptor_scale_param.sigValueChanged.connect(_scale_changed)

        receptor_parameters = pf.make_group(
            "Receptor layer",
            [
                receptor_scale_param,
                pf.make_bool("greyscale", readonly=True),
            ],
        )

        self.dock.add_parameter(receptor_parameters)

        # Horizontal cell parameters
        # ==================================================
        horizontal_parameters = pf.make_group(
            "Horizontal layer",
            [
                pf.make_int("Sectors", 96),
                pf.make_rf_params(
                    rf_params={
                        "create_feedback": True,
                        "kernel_params": {
                            "min_size": 3,
                        },
                    },
                ),
            ],
        )

        self.dock.add_parameter(horizontal_parameters)

        # Bipolar cell parameters
        # ==================================================
        bipolar_parameters = pf.make_group(
            "Bipolar layer",
            [
                pf.make_int("Sectors", 96),
                pf.make_rf_params(),
            ],
        )

        self.dock.add_parameter(bipolar_parameters)

        # Amacrine cell parameters
        # ==================================================
        amacrine_parameters = pf.make_group(
            "Amacrine layer",
            [
                pf.make_int("Sectors", 96),
                pf.make_rf_params(sectors=96),
            ],
        )

        self.dock.add_parameter(amacrine_parameters)

        # Ganglion cell parameters
        # ==================================================
        ganglion_parameters = pf.make_group(
            "Ganglion layer",
            [
                pf.make_int("Sectors", 96),
                pf.make_rf_params(sectors=96),
            ],
        )

        self.dock.add_parameter(ganglion_parameters)

        # If any settings change, we need to rerun everything.
        # This signal ensures that.
        self.dock.root.sigTreeStateChanged.connect(self.tree_changed)

        ch = pf.to_dict(self.dock.root, titles=False)

        pp(ch)
