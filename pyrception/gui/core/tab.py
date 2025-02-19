# --------------------------------------
import typing as tp

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QToolBar
from PySide6.QtWidgets import QDockWidget
from PySide6.QtWidgets import QProgressBar

# --------------------------------------
from pathlib import Path

# --------------------------------------
from pprint import pp

# --------------------------------------
from pyqtgraph.parametertree import Parameter

# --------------------------------------
from pyrception.conf import logger
from pyrception.visual import InputType
from pyrception.visual.utils.types import KernelFilter
from pyrception.utils.functions import thread_id
from pyrception.utils.functions import load_image
from pyrception.utils.functions import load_video
from pyrception.gui.core.canvas import Canvas
from pyrception.gui.core.dock import Dock
from pyrception.gui.core.splitview import SplitView
from pyrception.gui.param import factory as pf


class Tab(QMainWindow):
    create_layers = Signal()
    invalidate = Signal()

    def __init__(
        self,
        path: Path,
        itype: InputType,
        grayscale: bool = True,
        scale: float = None,
    ):
        super().__init__()

        logger.info(f" {thread_id()} ] Initialising tab...")

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

        # The layer stack
        # ==================================================
        self.create_layers.connect(self.canvas.create_layers)
        self.invalidate.connect(self.canvas.invalidate)

        # Status bar
        # ==================================================
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar(parent=self)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setGeometry(30, 40, 200, 25)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.setup_statusbar()

        # Slots & signals
        # ==================================================
        self.canvas.update_status.connect(self._update_status)
        # self.dock.sig_show_inactive_plots.connect(self.splitview._show_inactive_plots)
        # self.dock.sig_set_active_plot_colour.connect(
        #     self.splitview._set_neuron_plot_colour
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

    def _load(
        self,
        path: Path,
        itype: InputType,
        grayscale: bool = True,
        scale: float = None,
    ):

        if itype == InputType.Image:
            logger.info(f"Loading image '{path}'")
            self.canvas = Canvas(load_image(path, grayscale, scale), parent=self)

        elif itype == InputType.Video:
            logger.info(f"Loading video '{path}'")
            self.canvas = Canvas(
                load_video(path, grayscale, scale, probe=True), parent=self
            )

        self._update_view()

    def _update_view(self):
        if self.canvas is not None:
            self.splitview = SplitView(self.canvas)
            self.splitview.setSizes((2, 1))
            self.setCentralWidget(self.splitview)

    def setup_statusbar(self):
        self._update_status()

    @Slot(str, int)
    def _update_status(
        self,
        message: str = "Ready",
        value: int = 0,
    ):

        self.status_bar.showMessage(message)

        self.progress_bar.setValue(value)
        if value > 0:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

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
        act_scale_image.triggered.connect(
            lambda: self.canvas._auto_range(list(self.canvas.ivs.values()))
        )
        self.toolbar.addAction(act_scale_image)

        # Create retina layers
        act_create_layers = QAction(
            QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart),
            "Create layers",
            self.toolbar,
        )
        act_create_layers.triggered.connect(self._create_layers)
        self.toolbar.addAction(act_create_layers)

    @Slot()
    def _toggle_dock(self):
        self.dock.setVisible(not self.dock.isVisible())

    def _create_layers(self):
        self.status_bar.showMessage(f"Creating layers...")
        self.progress_bar.show()
        self.canvas.create_layers(self.dock.root)

    @Slot()
    def set_canvas(self):

        self.status_bar.showMessage(f"Setting up canvas...")
        self.progress_bar.setValue(0)
        self.canvas.set_stack(self.stack, auto_range=True)
        self.status_bar.clearMessage()

    @Slot()
    def tree_changed(self):

        self.ready = False
        self.invalidate.emit()

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
                            "filter": KernelFilter.Gaussian,
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
                pf.make_int("Sectors", 72),
                pf.make_rf_params(),
            ],
        )

        self.dock.add_parameter(bipolar_parameters)

        # Amacrine cell parameters
        # ==================================================
        amacrine_parameters = pf.make_group(
            "Amacrine layer",
            [
                pf.make_int("Sectors", 36),
                pf.make_rf_params(),
            ],
        )

        self.dock.add_parameter(amacrine_parameters)

        # Ganglion cell parameters
        # ==================================================
        ganglion_parameters = pf.make_group(
            "Ganglion layer",
            [
                pf.make_int("Sectors", 96),
                pf.make_float("Inhibition scale", 2.0),
                pf.make_rf_params(
                    name="bipolar_params",
                    title="Bipolar parameters",
                ),
                pf.make_rf_params(
                    name="amacrine_params",
                    title="Amacrone parameters",
                ),
            ],
        )

        self.dock.add_parameter(ganglion_parameters)

        # If any settings change, we need to rerun everything.
        # This signal ensures that.
        self.dock.root.sigTreeStateChanged.connect(self.tree_changed)

        pp(pf.to_dict(self.dock.root))