import pyqtgraph as pg


class Plot(pg.PlotWidget):
    # TODO: More fancy plotting
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
