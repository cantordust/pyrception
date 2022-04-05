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
    A group of ON- or OFF-type RGCs.
    """

    def __init__(
        self,
        _width: int,
        _height: int,
        _off: bool = False,
    ):

        # Bipolar cell layer.
        self.bplayer = BipolarLayer()

        # Comparison operator
        self.comp_op = pt.lt if _off else pt.gt

        # Actual activations
        self.activations = pt.zeros(_width)

        # A uniform baseline used in the element-wise
        # comparison with pt.where
        self.baseline = pt.zeros(_width)

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
        center_weight = 1.0 / (center_size ** 2)
        surround_weight = -1.0 / (_size ** 2 - center_size ** 2)

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

    def get_max_kernel_size(
        self,
        _init_ksize: int,
        _dilation: int,
        _height: int,
        _width: int,
        _scale: int = 1,
        _cover: bool = False,
    ):
        _sum = _init_ksize
        _short_fit_sum = 0
        _long_fit_sum = 0

        short_side = min(_height, _width)
        long_side = max(_height, _width)

        # Find the optimal coverage for the short and long sides
        ksize = _init_ksize
        while True:
            if _sum <= short_side:
                _short_fit_sum = _sum
            _long_fit_sum = _sum
            _sum += 2 * _scale * ksize * _dilation

            if _sum > long_side:
                break

            ksize += _dilation

        # print(f'==[ _short_fit_sum: {_short_fit_sum}')
        # print(f'==[ _long_fit_sum: {_long_fit_sum}')

        # 0 mask along the height
        hmask = (
            _height - _short_fit_sum
            if short_side == _height
            else _height - _long_fit_sum
        ) % ksize

        # 0 mask along the width
        wmask = (
            _width - _short_fit_sum if short_side == _width else _width - _long_fit_sum
        ) % ksize

        # print(f'==[ _ksize: {_ksize}')
        # print(f'==[ hmask: {hmask}')
        # print(f'==[ wmask: {wmask}')

        return (ksize, hmask, wmask)
