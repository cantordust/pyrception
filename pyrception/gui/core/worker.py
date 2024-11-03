# --------------------------------------
import typing as tp

# --------------------------------------
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtCore import Signal
from PySide6.QtCore import QEvent
from PySide6.QtCore import QObject
from PySide6.QtGui import QKeyEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtGui import QEnterEvent
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QSizePolicy

# --------------------------------------
from functools import partial

# --------------------------------------
from pyrception.conf import logger
from pyrception.utils.functions import thread_id


class Worker(QObject):

    finished = Signal()

    def __init__(
        self,
        pfun: tp.List[partial] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        logger.info(f"Worker initialised in thread {thread_id()}")
        if pfun is None:
            pfun = []
        self.pfun = pfun

    @Slot()
    def run(
        self,
        *args,
        **kwargs,
    ):
        logger.info(f"Worker running in thread {thread_id()}")
        if self.pfun is not None:
            for pf in self.pfun:
                pf()
        self.finished.emit()

    def bind(
        self,
        fun: tp.Callable,
        *args,
        **kwargs,
    ) -> partial:
        self.pfun.append(partial(fun, *args, **kwargs))

    def clear(self):
        self.pfun.clear()