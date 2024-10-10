import typing as tp

# --------------------------------------
from PySide6 import QtGui
from PySide6.QtGui import QPainter

# --------------------------------------
import pyqtgraph as pg
from pyqtgraph import functions as pgfn


class RadialLine(pg.ROI):
    """
    ROI subclass with two freely-moving handles defining a line.
    """

    def __init__(
        self,
        positions=(None, None),
        pos=None,
        **args,
    ):
        if pos is None:
            pos = [0, 0]

        self.endpoints = [pg.Point(p) for p in positions]
        pg.ROI.__init__(self, pos, [1, 1], **args)
        if len(positions) > 2:
            raise Exception(
                "LineROI must be defined by exactly 2 positions. For more points, use PolyLineROI."
            )

    def paint(self, p: QPainter, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.drawLine(self.endpoints[0], self.endpoints[1])

    def listPoints(self):
        return self.endpoints

    def getState(self):
        state = pg.ROI.getState(self)
        state["points"] = self.endpoints
        return state

    def saveState(self):
        state = pg.ROI.saveState(self)
        state["points"] = [tuple(p) for p in self.endpoints]
        return state

    def setState(self, state):
        pg.ROI.setState(self, state)
        p1 = [
            state["points"][0][0] + state["pos"][0],
            state["points"][0][1] + state["pos"][1],
        ]
        p2 = [
            state["points"][1][0] + state["pos"][0],
            state["points"][1][1] + state["pos"][1],
        ]
        self.movePoint(self.endpoints[0], p1, finish=False)
        self.movePoint(self.endpoints[1], p2)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QtGui.QPainterPath()

        ep1 = self.endpoints[0]
        ep2 = self.endpoints[1]
        dh = ep2 - ep1
        if dh.length() == 0:
            return p
        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p

        pxv *= 4

        p.moveTo(ep1 + pxv)
        p.lineTo(ep2 + pxv)

        return p

    def set_end(
        self,
        pos: tp.Tuple[float, float],
    ):
        self.endpoints[-1] = pg.Point(pos)

    def getArrayRegion(
        self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds
    ):
        """
        Use the position of this ROI relative to an imageItem to pull a slice
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value
        will have ndim = data.ndim-1

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        imgPts = [self.mapToItem(img, ep) for ep in self.endpoints]

        d = pg.Point(imgPts[1] - imgPts[0])
        o = pg.Point(imgPts[0])
        rgn = pgfn.affineSlice(
            data,
            shape=(int(d.length()),),
            vectors=[pg.Point(d.norm())],
            origin=o,
            axes=axes,
            order=order,
            returnCoords=returnMappedCoords,
            **kwds,
        )

        return rgn
