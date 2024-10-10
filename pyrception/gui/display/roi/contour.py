import typing as tp

# --------------------------------------
import pyqtgraph as pg


class Contour(pg.EllipseROI):
    """
    Circular contour subclass.
    """

    def __init__(self, pos, radius, **args):

        args.setdefault("movable", False)
        args.setdefault("resizable", False)
        args.setdefault("removable", False)
        args.setdefault("rotatable", False)

        pg.EllipseROI.__init__(
            self,
            pos,
            1,
            aspectLocked=True,
            **args,
        )

        for h in self.getHandles():
            self.removeHandle(h)

        self.setSize(2 * radius, (0.5, 0.5), update=True, finish=True)
