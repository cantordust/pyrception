import typing as tp

# --------------------------------------
import pyqtgraph as pg


class Target(pg.TargetItem):
    def __init__(
        self,
        pos: tp.Tuple,
        *args,
        **kwargs,
    ):

        kwargs.setdefault('movable', False)
        pg.TargetItem.__init__(
            self,
            pos,
            *args,
            **kwargs,
        )
