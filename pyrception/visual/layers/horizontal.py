from typing import *

# --------------------------------------
from pyrception.visual.layers.proto import ProtoLayer


class HorizontalLayer(ProtoLayer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        kwargs.setdefault("name", "Horizontal")
        super().__init__(*args, **kwargs)
        self.log("Initialised.")
