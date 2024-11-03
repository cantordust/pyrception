from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot

from PySide6.QtGui import QColor

from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QHeaderView
from PySide6.QtWidgets import QDockWidget
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QSizePolicy

# --------------------------------------
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree import ParameterTree

# --------------------------------------
from pyrception.gui.param import factory as pf
# from pyrception.gui.utils.functions import get_colour

class Dock(QDockWidget):
    sig_show_inactive_plots = Signal()
    sig_set_active_plot_colour = Signal()
    sig_set_inactive_plot_colour = Signal()
    sig_set_stack_contour_colour = Signal()
    sig_set_slice_contour_colour = Signal()
    sig_show_stack = Signal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ptree = ParameterTree(showHeader=True)
        self.root = pf.make_group("Root", expanded=True)
        self.setWidget(self.ptree)
        self.setMinimumWidth(300)
        self.ptree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.ptree.header().resizeSection(0, 200)
        self.ptree.addParameters(self.root, showTop=False)


    def add_parameter(
        self,
        parameter: Parameter,
    ):

        self.root.addChild(parameter)
