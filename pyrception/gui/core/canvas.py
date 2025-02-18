# --------------------------------------
import typing as tp

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtCore import Signal
from PySide6.QtCore import QThread
from PySide6.QtCore import QObject
from PySide6.QtGui import QKeyEvent
from PySide6.QtGui import QEnterEvent
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QSizePolicy

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
from pyqtgraph.parametertree import Parameter

# --------------------------------------
from pyrception.conf import logger
from pyrception.utils.functions import thread_id
from pyrception.gui.param import factory as pf
from pyrception.gui.core.image import ImageView
from pyrception.gui.core.worker import Worker
from pyrception.visual.layers import ReceptorLayer
from pyrception.visual.layers import HorizontalLayer
from pyrception.visual.layers import BipolarLayer
from pyrception.visual.layers import AmacrineLayer
from pyrception.visual.layers import GanglionLayer

# from pyrception.gui import Layer
# from pyrception.gui import Stack
# from pyrception.gui.conf import conf
# from pyrception.gui.core.roi.contour import Contour
# from pyrception.gui.core.roi.target import Target


class Canvas(QWidget):
    plot = Signal()
    highlight_plot = Signal(int)
    # update_radial_plot = Signal(float)
    # update_phase_plot = Signal(float)
    update_status = Signal(str, int)
    start_worker = Signal()

    class EventHandler(QObject):

        ctrl_signal = Signal(bool)
        focus_signal = Signal()

        def __init__(self, wh, *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.wh = wh

        def eventFilter(self, obj: tp.Any, event: tp.Any):
            if obj is self.wh:
                if isinstance(event, QEnterEvent):
                    self.focus_signal.emit()
                elif isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Control:
                    if event.type() == QKeyEvent.Type.KeyPress:
                        self.ctrl_signal.emit(True)
                    elif event.type() == QKeyEvent.Type.KeyRelease:
                        self.ctrl_signal.emit(False)
            return QObject().eventFilter(obj, event)

    def __init__(
        self,
        frames: np.ndarray = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Mouse tracking for the ROI
        # ==================================================
        self.mouse_tracking_toggle = False
        # self.mouse_tracker = SignalProxy(
        #     self.iv.scene.sigMouseMoved,
        #     rateLimit=100,
        #     slot=self._mouse_coordinates,
        # )

        self.event_handler = Canvas.EventHandler(self)
        self.installEventFilter(self.event_handler)
        # self.event_handler.ctrl_signal.connect(self._toggle_mouse_tracking)
        # self.event_handler.focus_signal.connect(self._focus_canvas)

        # Images
        # ==================================================
        self.frames = frames

        # Thread for preventing the GUI from blocking
        # ==================================================
        self._thread = QThread()
        self._worker = Worker()
        self.start_worker.connect(self._worker.run)
        self._worker.moveToThread(self._thread)
        self._thread.start()

        # Image viewport
        # ==================================================
        self.ivs = {}

        if self.frames is not None:
            self.ivs["original"] = ImageView(self.frames, True)
            self.ivs["receptor"] = ImageView(np.zeros_like(self.frames))
            self.ivs["horizontal"] = ImageView(np.zeros_like(self.frames))
            self.ivs["normalised"] = ImageView(np.zeros_like(self.frames))
            self.ivs["bipolar"] = ImageView(np.zeros_like(self.frames))
            self.ivs["amacrine"] = ImageView(np.zeros_like(self.frames))
            self.ivs["ganglion"] = ImageView(np.zeros_like(self.frames))

        # Layout grid
        # ==================================================
        self.grid = QGridLayout(self)

        self.grid.addWidget(self.ivs["original"], 0, 0, 1, 1)
        self.grid.addWidget(self.ivs["receptor"], 0, 1, 1, 1)
        self.grid.addWidget(self.ivs["horizontal"], 0, 2, 1, 1)
        self.grid.addWidget(self.ivs["normalised"], 1, 0, 1, 1)
        self.grid.addWidget(self.ivs["bipolar"], 1, 1, 1, 1)
        self.grid.addWidget(self.ivs["amacrine"], 2, 0, 1, 1)
        self.grid.addWidget(self.ivs["ganglion"], 2, 1, 1, 1)

        # Layers
        # ==================================================
        self.receptor = None
        self.horizontal = None
        self.bipolar = None
        self.amacrine = None
        self.ganglion = None

        # Thumbnails
        # ==================================================
        # self.thumbnails = []
        # self.tb_widget = QWidget()
        # self.tb_scroll_area = QScrollArea(self)
        # self.tb_layout = QHBoxLayout()
        # self.tb_layout.setContentsMargins(0, 0, 0, 0)
        # self.tb_widget.setLayout(self.tb_layout)

    def __del__(self):

        print(f"Quitting thread...")
        # if self._thread.isRunning():
        #     self._thread.quit()
        #     while self._thread.isRunning():
        #         time.sleep(0.1)
        #         print(f"Quitting thread...")

    def _auto_range(
        self,
        iviews: list[ImageView],
    ):
        for iview in iviews:
            iview.autoRange()

    def invalidate(self):
        """
        A method that invalidates the current layers.

        This forces all layers to be recreated.
        """
        for iview in self.ivs:
            logger.info

    # @property
    # def layer(self) -> Layer:
    #     return self.stack.layers[self.stack.current_layer]

    # def set_stack(
    #     self,
    #     stack: Stack,
    #     auto_range: bool = False,
    # ):
    #     # Update the stack
    #     # ==================================================
    #     self.stack = stack
    #     if len(self.stack.layers) > 0:
    #         self.stack._update_current_layer()

    #     # Update the thumbnails
    #     # ==================================================
    #     self._update_thumbnails()

    #     # Select the first layer
    #     # ==================================================
    #     self.select_layer(0, auto_range)

    # @Slot(int, bool)
    # def select_layer(
    #     self,
    #     layer: int,
    #     auto_range: bool = False,
    # ):
    #     cur_layer = self.stack.current_layer
    #     if cur_layer is None:
    #         cur_layer = 0

    #     # Update the layer
    #     # ==================================================
    #     self.stack._update_current_layer(layer)

    #     # Highlight the selected thumbnail
    #     # ==================================================
    #     self.thumbnails[cur_layer].deselect()
    #     self.thumbnails[layer].select()

    #     # Process the layer
    #     self.process(auto_range)

    #     # Emit a signal to highlight the relevant plots
    #     # ==================================================
    #     # with np.printoptions(threshold=np.inf):
    #     #     print(f"==[ layer phase profile: {self.layer.phase_profile}")

    #     self.highlight_plot.emit(layer)

    # def process(
    #     self,
    #     auto_range: bool = False,
    # ):
    #     # Paint the result onto the canvas
    #     # ==================================================
    #     self._draw(auto_range)

    #     # Set up the ROI
    #     # ==================================================
    #     self.iv.set_roi(self.layer)

    #     # Plot the radial and phase profiles
    #     # ==================================================
    #     self.plot.emit()

    # def _draw(
    #     self,
    #     auto_range: bool = False,
    # ):
    #     # Coordinates of the current layer and the stack
    #     # ==================================================
    #     lcx, lcy = self.layer.centre
    #     scx, scy = self.stack.centre

    #     # Reset the canvas
    #     # ==================================================
    #     self.image = np.zeros(self.stack.merged.shape + (4,))

    #     # Draw the slice and potentially the stack
    #     # ==================================================
    #     if conf.show_stack:
    #         idx = np.argwhere(self.stack.merged > 0).T
    #         self.image[idx[0], idx[1], 0] = conf.stack_contour_colour.red()
    #         self.image[idx[0], idx[1], 1] = conf.stack_contour_colour.green()
    #         self.image[idx[0], idx[1], 2] = conf.stack_contour_colour.blue()
    #         self.image[idx[0], idx[1], 3] = conf.stack_contour_colour.alpha()

    #     idx = np.argwhere(self.layer.image > 0).T
    #     self.image[idx[0], idx[1], 0] = conf.slice_contour_colour.red()
    #     self.image[idx[0], idx[1], 1] = conf.slice_contour_colour.green()
    #     self.image[idx[0], idx[1], 2] = conf.slice_contour_colour.blue()
    #     self.image[idx[0], idx[1], 3] = conf.slice_contour_colour.alpha()

    #     # Draw the centre
    #     # ==================================================
    #     if self.slice_centre is not None:
    #         self.iv.removeItem(self.slice_centre)
    #     self.slice_centre = Target(
    #         (lcy + 0.5, lcx + 0.5),
    #         pen=self.slice_centre_pen,
    #     )
    #     self.iv.addItem(self.slice_centre)

    #     # Draw the slice contour
    #     # ==================================================
    #     if self.slice_contour is not None:
    #         self.iv.removeItem(self.slice_contour)
    #     self.slice_contour = Contour(
    #         (lcy + 0.5, lcx + 0.5),
    #         radius=self.layer.radius,
    #         pen=self.slice_contour_pen,
    #     )
    #     self.iv.addItem(self.slice_contour)

    #     # Draw the stack contour
    #     # ==================================================
    #     if self.stack_contour is not None:
    #         self.iv.removeItem(self.stack_contour)
    #     self.stack_contour = Contour(
    #         (scy + 0.5, scx + 0.5),
    #         radius=self.stack.radius,
    #         pen=self.stack_contour_pen,
    #     )
    #     self.iv.addItem(self.stack_contour)

    #     # Set the image
    #     # ==================================================
    #     self.iv.setImage(
    #         self.image, autoRange=auto_range, levels=(0, 255), levelMode="rgba"
    #     )

    # def _update_thumbnails(self):
    #     while True:
    #         item = self.tb_layout.takeAt(0)
    #         if item is None or item.isEmpty():
    #             break
    #         self.tb_layout.removeWidget(item.widget())
    #         item.widget().deleteLater()

    #     self.thumbnails.clear()
    #     for index, layer in enumerate(self.stack.layers):
    #         tb = Thumbnail(index, layer, self, 90)
    #         tb._selected.connect(self.select_layer)
    #         self.thumbnails.append(tb)

    #     for idx, tb in enumerate(self.thumbnails):
    #         self.tb_layout.addWidget(tb, alignment=Qt.AlignmentFlag.AlignLeft)

    #     self.tb_layout.addStretch()

    @Slot(float)
    def scale_source(
        self,
        scale: float,
    ):

        tr = QTransform()
        tr.scale(scale, scale)
        self.ivs["original"].imageItem.setTransform(tr)

    def _update_status(
        self,
        message: str = "Ready",
        value: int = 0,
    ):
        self.update_status.emit(message, value)

    def add_receptor(
        self,
        shape: tuple,
        **params,
    ):

        logger.info(f"[ {thread_id()} ] Creating receptor layer...")

        self._update_status("Creating receptor layer...")

        self.receptor = ReceptorLayer(
            shape,
            notifier=self._update_status,
            **params,
        )

        self._update_status("Creating receptor layer...done!")

    def add_horizontal(
        self,
        shape: tuple,
        **params,
    ):

        logger.info(f"[ {thread_id()} ] Creating horizontal layer...")
        self._update_status("Creating horizontal layer...")

        self.horizontal = HorizontalLayer(
            shape,
            self.receptor,
            notifier=self._update_status,
            **params,
        )

        (_, _, _, canvas) = self.horizontal.plot_rfs()

        # canvas = ski.util.img_as_ubyte(canvas)

        self.ivs["horizontal"].setImage(canvas)
        self.ivs["horizontal"].autoRange()
        self._update_status("Creating horizontal layer...done!")

    def add_bipolar(
        self,
        shape: tuple,
        **params,
    ):

        self._update_status("Creating bipolar layer...")
        self.bipolar = BipolarLayer(
            shape,
            self.receptor,
            self.horizontal,
            notifier=self._update_status,
            **params,
        )

        (_, _, _, canvas) = self.bipolar.plot_rfs()

        # canvas = ski.util.img_as_ubyte(canvas)

        self.ivs["bipolar"].setImage(canvas)
        self.ivs["bipolar"].autoRange()
        self._update_status("Creating bipolar layer...done!")

    def add_amacrine(
        self,
        shape: tuple,
        **params,
    ):

        self._update_status("Creating amacrine layer...")
        self.amacrine = AmacrineLayer(
            shape,
            self.bipolar,
            notifier=self._update_status,
            **params,
        )

        (_, _, _, canvas) = self.amacrine.plot_rfs()

        # canvas = ski.util.img_as_ubyte(canvas)

        self.ivs["amacrine"].setImage(canvas)
        self.ivs["amacrine"].autoRange()
        self._update_status("Creating amacrine layer...done!")

    def add_ganglion(
        self,
        shape: tuple,
        **params,
    ):

        self._update_status("Creating ganglion layer...")
        self.ganglion = GanglionLayer(
            shape,
            self.bipolar,
            self.amacrine,
            notifier=self._update_status,
            **params,
        )

        (_, _, _, canvas) = self.ganglion.plot_rfs()

        # canvas = ski.util.img_as_ubyte(canvas)

        self.ivs["ganglion"].setImage(canvas)
        self.ivs["ganglion"].autoRange()
        self._update_status("Creating ganglion layer...done!")

    @Slot(Parameter)
    def create_layers(
        self,
        root: Parameter,
    ):

        params = pf.to_dict(root)
        shape = (
            np.array(
                [
                    params["input"]["height"],
                    params["input"]["width"],
                ]
            )
        )

        self._worker.clear()
        self._worker.bind(self.add_receptor, shape, **params["receptor_layer"])
        self._worker.bind(self.add_horizontal, shape, **params["horizontal_layer"])
        self._worker.bind(self.add_bipolar, shape, **params["bipolar_layer"])
        self._worker.bind(self.add_amacrine, shape, **params["amacrine_layer"])
        self._worker.bind(self.add_ganglion, shape, **params["ganglion_layer"])
        self._worker.bind(self._update_status)
        self.start_worker.emit()

    # def eventFilter(self, obj, event):
    #     if obj is self.window:
    #         if event.type() == QEvent.KeyPress:
    #             if event.key() == Qt.Key_Control:
    #                 self.ctrl_signal.emit(True)
    #         if event.type() == QEvent.KeyRelease:
    #             if event.key() == Qt.Key_Control:
    #                 self.ctrl_signal.emit(False)
    #     return super().eventFilter(obj, event)

    # @Slot()
    # def _mouse_coordinates(
    #     self,
    #     event: MouseClickEvent,
    # ):
    #     if self.mouse_tracking_toggle:
    #         if self.iv.radial_roi is not None:
    #             pos = self.iv.getView().vb.mapSceneToView(event)[0]

    #             cx = pos.x() - self.layer.centre[1] - 0.5
    #             cy = pos.y() - self.layer.centre[0] - 0.5
    #             r = np.sqrt(cx**2 + cy**2)
    #             radius = r / self.layer.radius
    #             r2 = 2 * r
    #             if radius <= 1:
    #                 if r == 0:
    #                     phase = 0
    #                 else:
    #                     phase = self.rad2deg * np.arccos(cx / r)
    #                 if cy < 0:
    #                     phase = 360 - phase

    #                 self.iv.radial_roi.setSize(
    #                     r2,
    #                     center=(0.5, 0.5),
    #                     update=True,
    #                     finish=True,
    #                 )
    #                 self.iv.phase_roi.set_end(pos)
    #                 self.update_radial_plot.emit(radius)
    #                 self.update_phase_plot.emit(phase)

    # @Slot(bool)
    # def _toggle_mouse_tracking(
    #     self,
    #     enable: bool,
    # ):
    #     self.mouse_tracking_toggle = enable

    # @Slot()
    # def _focus_canvas(self):
    #     self.setFocus()

    # @Slot()
    # def _set_slice_contour_colour(self):
    #     self.slice_contour_pen = pg.mkPen(color=conf.slice_contour_colour, width=1)
    #     self._draw()

    # @Slot()
    # def _set_stack_contour_colour(self):
    #     self.stack_contour_pen = pg.mkPen(color=conf.stack_contour_colour, width=1)
    #     self._draw()

    # @Slot()
    # def _set_show_stack(self):
    #     self._draw()
