# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Optional

# --------------------------------------
import torch as pt
import torch.functional as ptf

# --------------------------------------
from pyrception.visual.bipolar import BipolarLayer


class GanglionLayer:

    """
    A layer of ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        width: int,
        height: int,
        off: bool = False,
    ):

        # Bipolar cell layer.
        self.bplayer = BipolarLayer()

        # Actual activations
        self.activations = pt.zeros(width)

        # A uniform baseline used in the element-wise
        # comparison with pt.where
        self.baseline = pt.zeros(width)

    def make_rf(_size: int):

        # Figure out if the kernel size is odd or even
        odd = _size % 2 == 1

        # Center size
        center_size = _size // 2

        if odd and center_size % 2 == 0:
            center_size += 1

        elif not odd and center_size % 2 == 1:
            center_size -= 1

        print(f"==[ center size: {center_size}")

        # Surround size
        surround_size = (_size - center_size) // 2
        print(f"==[ surround size:\n{surround_size}")

        # Center and surround receptor weights
        center_weight = 1.0 / (center_size**2)
        surround_weight = -1.0 / (_size**2 - center_size**2)

        # Create the center and surround
        center = pt.ones((center_size, center_size)) * center_weight

        kernel = ptf.pad(
            center,
            (surround_size, surround_size, surround_size, surround_size),
            "constant",
            surround_weight,
        )

        # Confirm that the kernel sums to 0 if illuminated uniformly
        # print(f'==[ kernel:\n{kernel}')
        # print(f'==[ sum: {kernel.sum()}')
        return kernel

    # kernel = make_rf(10)
    # print(f'==[ kernel: {kernel}')

    def _activate(self, norm_dev):

        # Get the normalised deviation from the mean.
        # The activation depends on whether the cells are ON or OFF.
        norm_dev = pt.where(
            self.comp_op(norm_dev, self.baseline),
            pt.abs(norm_dev),
            self.baseline,
        )

        # Compute the new activations.
        self.activations = pt.tanh(norm_dev)
