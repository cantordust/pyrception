import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot

from PySide6.QtWidgets import QSplitter

from PySide6.QtGui import QColor

# --------------------------------------
import pyqtgraph as pg

# --------------------------------------
# from pyrception.gui.conf import conf
from pyrception.gui.display.canvas import Canvas
# from pyrception.gui import Plot
# from pyrception.gui import Layer


class SplitView(QSplitter):
    def __init__(
        self,
        canvas: Canvas,
    ):
        super().__init__()

        # Canvas
        # ==================================================
        self.canvas = canvas
        self.addWidget(self.canvas)

    # @Slot(list, int)
    # def plot(
    #     self,
    #     layers: tp.List[Layer],
    #     current_layer_idx: int,
    # ):
    #     self.radial_plots.clear()
    #     self.phase_plots.clear()

    #     self.radial_graph.clear()
    #     self.phase_graph.clear()

    #     for radial_plot in self.radial_plots.values():
    #         self.radial_graph.removeItem(radial_plot)

    #     for phase_plot in self.phase_plots.values():
    #         self.phase_graph.removeItem(phase_plot)

    #     # Plot all layers
    #     # ==================================================
    #     for idx, layer in enumerate(layers):
    #         if idx == current_layer_idx:
    #             pen = self.active_pen
    #             self.current_layer_idx = idx
    #             self.current_layer = layer
    #             self.phase_graph.setXRange(0.0, layer.phase_range.max())
    #         else:
    #             if conf.show_inactive_plots:
    #                 pen = self.inactive_pen
    #             else:
    #                 pen = self.invisible_pen

    #         # Plot the radial profile
    #         if layer.radial_range is not None:
    #             self.radial_plots[idx] = self.radial_graph.plot(
    #                 layer.radial_range,
    #                 layer.radial_profile,
    #                 pen=pen,
    #             )

    #         # Plot the phase profile
    #         if layer.phase_range is not None:
    #             self.phase_plots[idx] = self.phase_graph.plot(
    #                 layer.phase_range,
    #                 layer.phase_profile,
    #                 pen=pen,
    #             )

    #         if idx == current_layer_idx:
    #             self.radial_arrow = pg.CurveArrow(self.radial_plots[idx])
    #             self.radial_arrow.setRotation(270)
    #             self.radial_arrow._rotate = False
    #             self.phase_arrow = pg.CurveArrow(self.phase_plots[idx])
    #             self.phase_arrow.setRotation(270)
    #             self.phase_arrow._rotate = False

    #     # Add vertical guides
    #     # ==================================================
    #     self.radial_graph.addItem(self.radial_guide)
    #     self.radial_graph.addItem(self.radial_arrow)
    #     self.phase_graph.addItem(self.phase_guide)
    #     self.phase_graph.addItem(self.phase_arrow)

    # @Slot(float, str)
    # def _update_radial_pos(
    #     self,
    #     radius: float,
    # ):
    #     # Find the index of the closest value in self.radial_xs
    #     xs = self.current_layer.radial_range
    #     diff = np.absolute(xs - radius)
    #     index = diff.argmin()
    #     x = xs[index]
    #     y = self.current_layer.radial_profile[index]
    #     self.radial_guide.setPos(x)
    #     self.radial_arrow.setIndex(index)
    #     self.radial_lbl.setText(f"{x.item():0.3f}|{y:0.3f}")

    # @Slot(float, str)
    # def _update_phase_pos(
    #     self,
    #     phase: float,
    # ):
    #     # Find the index of the closest phase value
    #     xs = self.current_layer.phase_range
    #     diff = np.absolute(xs - phase)
    #     index = diff.argmin()
    #     x = xs[index]
    #     y = self.current_layer.phase_profile[index]
    #     self.phase_guide.setPos(x)
    #     self.phase_arrow.setIndex(index)
    #     self.phase_lbl.setText(f"{x.item():3.3f}|{y:0.3f}")

    # @Slot()
    # def _show_inactive_plots(self):

    #     for idx in self.radial_plots:
    #         if idx != self.current_layer_idx:
    #             if conf.show_inactive_plots:
    #                 self.radial_plots[idx].setPen(self.inactive_pen)
    #                 self.phase_plots[idx].setPen(self.inactive_pen)
    #                 self.radial_plots[idx].update()
    #                 self.phase_plots[idx].update()
    #             else:
    #                 self.radial_plots[idx].setPen(self.invisible_pen)
    #                 self.phase_plots[idx].setPen(self.invisible_pen)
    #                 self.radial_plots[idx].update()
    #                 self.phase_plots[idx].update()

    # @Slot()
    # def _set_inactive_plot_colour(self):

    #     self.inactive_pen = pg.mkPen(
    #         color=conf.inactive_plot_colour,
    #         width=self.inactive_pen_width,
    #     )
    #     if conf.show_inactive_plots:
    #         for idx in self.radial_plots:
    #             if idx != self.current_layer_idx:
    #                 self.radial_plots[idx].setPen(self.inactive_pen)
    #                 self.radial_plots[idx].update()
    #                 self.phase_plots[idx].setPen(self.inactive_pen)
    #                 self.phase_plots[idx].update()

    # @Slot()
    # def _set_active_plot_colour(self):

    #     self.active_pen = pg.mkPen(
    #         color=conf.active_plot_colour,
    #         width=self.active_pen_width,
    #     )
    #     idx = self.current_layer_idx
    #     self.radial_plots[idx].setPen(self.active_pen)
    #     self.radial_plots[idx].update()
    #     self.phase_plots[idx].setPen(self.active_pen)
    #     self.phase_plots[idx].update()

    # @Slot(int)
    # def _highlight_plot(
    #     self,
    #     layer_idx: int,
    # ):

    #     self.radial_plots[self.current_layer_idx].setPen(self.inactive_pen)
    #     self.radial_plots[layer_idx].setPen(self.active_pen)
    #     self.radial_plots[layer_idx].update()
    #     self.radial_graph.getPlotItem().enableAutoRange()

    #     self.phase_plots[self.current_layer_idx].setPen(self.inactive_pen)
    #     self.phase_plots[layer_idx].setPen(self.active_pen)
    #     self.phase_plots[layer_idx].update()
    #     self.phase_graph.getPlotItem().enableAutoRange()

    #     self.current_layer_idx = layer_idx
