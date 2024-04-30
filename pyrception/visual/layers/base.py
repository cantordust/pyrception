from typing import *

# --------------------------------------
import numpy as np

# --------------------------------------
from pyrception.visual.util.types import Dim
from pyrception.visual.util.types import Dims
from pyrception.conf import logger


class BaseLayer:

    def __init__(
        self,
        name: str,
    ):

        self.name = name

    def debug(self, message: str):
        logger.debug(f"{self.name} | {message}")

    def info(self, message: str):
        logger.info(f"{self.name} | {message}")

    def warn(self, message: str):
        logger.warning(f"{self.name} | {message}")

    def error(self, message: str):
        logger.error(f"{self.name} | {message}")

    def _compute_dimensions(
        self,
        shape: Tuple[int],
        height: int = None,
        width: int = None,
        saccades: bool = False,
    ) -> Dims:
        """
        Compute various dimensions for the original and processed input.

        Args:
            scaled_height (Optional[int], optional):
                _description_. Defaults to None.

            scaled_width (Optional[int], optional):
                _description_. Defaults to None.

            saccades (bool, optional):
                _description_. Defaults to False.

        Returns:
            Dims: Dimensions.
        """

        dims = Dims(Dim(*shape), Dim(*shape))
        dims.comp.span = dims.comp.height * dims.comp.width * dims.comp.depth

        dims.resize = False

        if bool(width) and bool(height):
            raise ValueError("Please specify a scaled height *or* width, but not both!")

        elif bool(width) ^ bool(height):

            if height is not None:
                # Fixed height, calculate the width with the same AR
                pct = height / float(dims.orig.height)
                width = int((float(dims.orig.width) * pct))

            elif width is not None:
                # Fixed width, calculate the height with the same AR
                pct = width / float(dims.orig.width)
                height = int((float(dims.orig.height) * pct))

            dims.comp.height = height
            dims.comp.width = width
            dims.comp.span = dims.comp.height * dims.comp.width * dims.comp.depth
            dims.resize = True

        # Left and right padding
        lr_padding = (dims.comp.width // 2 + dims.comp.width % 2) if saccades else 0

        # Top and bottom padding
        tb_padding = (dims.comp.height // 2 + dims.comp.height % 2) if saccades else 0

        dims.padded.width = dims.comp.width + 2 * lr_padding
        dims.padded.height = dims.comp.height * dims.comp.depth + 2 * tb_padding
        dims.padded.depth = dims.comp.depth
        dims.padded.span = dims.padded.width * dims.padded.height * dims.padded.depth

        # The frame is padded only at the top and the bottom,
        # but the left and right padding values are used
        # to compute the size of the retinal field below.
        dims.padding = np.array(
            [
                lr_padding,
                lr_padding,
                tb_padding,
                tb_padding,
            ]
        )

        return dims
