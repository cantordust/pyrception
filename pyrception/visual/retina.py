# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import List
from typing import Tuple
from typing import Optional

# --------------------------------------
import torch as pt

# --------------------------------------
from pyrception.visual.ganglion import GanglionLayer


class Retina:

    """
    A retinal layer aims to emulate the operation of the retina
    with separate ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        _input_shape: Tuple[int, int],
        _tau: Optional[pt.Tensor] = None,
        *args,
        **kwargs,
    ):

        # The retinalActivation is graded.

        # Half of the RGCs are ON-type (activated by positive deviations from the mean input)
        # and the other half are OFF-type (activated by negative deviations from the mean input)

        if _tau is None:
            _tau = pt.full((_input_shape,), 100.0)

        super().__init__(_input_shape, _tau, *args, **kwargs)

        # ON RGCs
        self.on = GanglionLayer(_off=False, _size=_input_shape)

        # OFF RGCs
        self.off = GanglionLayer(_off=True, _size=_input_shape)

        # The size of the retinal layer is twice the size of the input
        self.activations = pt.zeros(2 * _input_shape)

    @staticmethod
    def get_max_kernel_size(
        _imh: int,
        _imw: int,
        _ksize: int,
        _dilation: int,
    ):
        _sum = _ksize
        _short_fit_sum = 0
        _long_fit_sum = 0

        # Short and long sides of the image
        short_side = min(_imh, _imw)
        long_side = max(_imh, _imw)

        # Find the optimal coverage for the short and long sides
        while True:
            if _sum <= short_side:
                _short_fit_sum = _sum
            _long_fit_sum = _sum
            _sum += 2 * _ksize * _dilation

            if _sum > long_side:
                break

            _ksize += _dilation

        # print(f'==[ _short_fit_sum: {_short_fit_sum}')
        # print(f'==[ _long_fit_sum: {_long_fit_sum}')

        # 0 mask along the height
        hmask = (
            _imh - _short_fit_sum if short_side == _imh else _imh - _long_fit_sum
        ) % _ksize

        # 0 mask along the width
        wmask = (
            _imw - _short_fit_sum if short_side == _imw else _imw - _long_fit_sum
        ) % _ksize

        # print(f'==[ _ksize: {_ksize}')
        # print(f'==[ hmask: {hmask}')
        # print(f'==[ wmask: {wmask}')

        return (_ksize, hmask, wmask)

    @staticmethod
    def make_edrf(
        _ih: int,
        _iw: int,
        _rscale: int = 1,
    ):
        """
        Eccentricity-dependent Receptive Fields.
        """
        # Size of the image unrolled into a vector
        isize = _ih * _iw

        # 9x9 kernel
        ksize = 9

        rowidx = []
        colidx = []
        values = []

        for ks in range(ksize):
            rowidx.extend([1] * ksize)
            colidx.extend([i for i in range(ksize)])
            values.extend([1.0] * ksize)

        edrf = pt.sparse_coo_tensor([rowidx, colidx], values, size=(_ih, _iw))

        print(f"==[ edrf:\n{edrf.to_dense()}")

    def _interleave(self):

        """
        Interleave on/off activations.
        """

        self.activations = pt.reshape(
            pt.stack((self.on.activations, self.off.activations), axis=1),
            (-1,),
        )

        if self.activation_history is not None:
            self.activation_history.append(self.activations.squeeze_().numpy())

    def integrate(
        self,
        _input_signals: pt.Tensor,
    ):

        """
        Integrate raw input signals and trigger action potentials.
        """

        # Integrate input signals.
        self.potentials = _input_signals

        # Get the normalised deviation
        norm_dev = self._norm_deviation()

        # Update the stats
        self._update_potential_stats()

        # Compute the activations of ON and OFF cells
        self.on._activate(norm_dev)
        self.off._activate(norm_dev)

        # Interleave the activations of on and off cells.
        self._interleave()

    @staticmethod
    def _make_norm_rfields(
        _ksize: int,
        _dilation: int = 2,
        _scale: int = 1,
        _kernels: Optional[List] = None,
    ):

        # Stretch
        s = 2 * _scale * _dilation

        # Repetitions per kernel
        repetitions = []

        # Foveal region
        if len(_kernels) > 0:
            # Number of times to repeat the kernel
            n = _ksize - _dilation
            retina = _kernels[0].repeat(_scale * 2 * (n + s), _scale * 2 * (n + s))

            # Update the number of kernel repetitions
            repetitions.append(_scale * 2 * (n + s))

        print(f"==[ fovea size: {retina.size()}")

        # The rest of the retina
        for k_i, kernel in enumerate(_kernels[1:]):

            # Number of times to repeat the kernel
            n = _ksize

            # Update the repetitions
            repetitions.append(_scale * 2 * (n + s))

            # Increment the kernel size
            _ksize += _dilation

            # Vertical repetition
            vrep = kernel.repeat(_scale * s, _scale * 2 * n)
            # print(f'==[ vpad size: {vpad.size()}')

            # Horizontal repetition
            hrep = kernel.repeat(_scale * 2 * (n + s), _scale * s)
            # print(f'==[ hpad size: {hpad.size()}')

            # print(f'==[ retina size: {retina.size()}')
            retina = pt.cat((vrep, retina), 0)
            retina = pt.cat((retina, vrep), 0)
            # print(f'==[ retina size: {retina.size()}')

            retina = pt.cat((hrep, retina), 1)
            retina = pt.cat((retina, hrep), 1)
            print(f"==[ retina size: {retina.size()}")

            # print(f'==[ patch:\n{patch}')

        print(f"==[ repetitions: {repetitions}")

        return (retina, repetitions)

    @staticmethod
    def _make_retinotopic_grid(
        _ih: int,
        _iw: int,
        _init_ksize: int = 3,
        _dilation: int = 2,
    ):
        # Compute the maximal kernel size
        _max_ksize = Retina.get_max_kernel_size(_init_ksize, _dilation, _ih, _iw)

        # Store the kernels in a list
        kernels = []

        for _ks in range(_init_ksize, _max_ksize + 1, _dilation):
            # print(f"==[ kernel: {_ks} x {_ks}")
            kernels.append(GanglionLayer.make_rf(_ks))
