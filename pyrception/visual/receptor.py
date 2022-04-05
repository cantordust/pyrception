# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
import enum

# --------------------------------------
from dotmap import DotMap

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf
from torch.distributions.one_hot_categorical import OneHotCategorical

pt.set_printoptions(edgeitems=20)


# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception.aux.types import View


class ReceptorLayer:

    """
    A layer of receptors and horizontal cells.

    This layer applies local normalisation to the input
    by looking at an eccentricity-dependent patch of receptors
    and their activations to normalise the activation of the
    receptor in the centre.
    """

    def __init__(
        self,
        _source: Union[np.ndarray, cv.VideoCapture],
        _scaled_height: Optional[int] = None,
        _scaled_width: Optional[int] = None,
        _mode: Optional[enum.Enum] = cv.COLOR_RGB2GRAY,
        _k_max: int = 15,
        _k_min: int = 3,
        _sd_w: int = 1 / 4,
        _sd_h: int = 1 / 4,
        *args,
        **kwargs,
    ):

        # Store the source
        self.source = _source

        # Compute source and output dimensions
        if _scaled_height is not None and _scaled_width is not None:
            raise ValueError("Please provide the new height *or* width, but not both.")

        # Flag indicating whether the source is still being processed
        self.processing = True

        # Current frame
        self.frame = None

        # A function used to extract a source frame
        if not isinstance(_source, (np.ndarray, cv.VideoCapture)):
            raise TypeError(f"Invalid input source type '{type(_source)}'.")

        elif isinstance(_source, cv.VideoCapture):
            self.frame_op = lambda processing, src: src.read()

        if isinstance(_source, np.ndarray):
            self.frame = _source
            self.frame_op = lambda processing, src: (processing, self.frame)

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(_scaled_height, _scaled_width)

        # Create the receptor field.
        # This is twice the size of the
        # original input in each dimension.
        self.rf, self.norm_mask = self._make_rf(
            _k_max,
            _k_min,
            _sd_w,
            _sd_h,
        )

        # Create a flatmask
        self.flatmask = self._make_flatmask(_mode)

        # Change the depth of the processed frame.
        if self.flatmask is not None:
            self.dim.D = 1

        # Display some useful info
        print(f"==[ Press ESC to quit.")

    def __del__(self):

        self.processing = False
        cv.destroyAllWindows()

        if isinstance(self.source, cv.VideoCapture):
            self.source.release()

    def _get_frame(
        self,
        _probe: Optional[bool] = False,
    ):
        """
        Extract a frame at a certain offset from the center.
        """

        # Get the next frame by applying self.frame_op to the source
        self.processing, frame = self.frame_op(self.processing, self.source)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        if _probe:
            self.processing = True
            return frame

        if self.dim.resize:
            frame = cv.resize(
                frame,
                (self.dim.W, self.dim.H),
                interpolation=cv.INTER_AREA,
            )

        frame = pt.from_numpy(frame).float()

        return frame

    def _make_flatmask(
        self,
        _mode: Optional[enum.Enum] = None,
    ):
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        if self.dim.orig.D == 1 or _mode is None:
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.dim.H, self.dim.W, self.dim.D))

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = OneHotCategorical(probs)

        return ohc.sample()

        # print(f"==[ R: {self.flatmask[:,:,0].sum()}")
        # print(f"==[ G: {self.flatmask[:,:,1].sum()}")
        # print(f"==[ B: {self.flatmask[:,:,2].sum()}")

        # # Create the actual tensor
        # self.st = pt.sparse_coo_tensor(
        #     np.array(
        #         [
        #             sparse_rows,
        #             sparse_cols,
        #         ]
        #     ),
        #     np.array(sparse_vals),
        #     size=(st_size, st_size),
        # ).to_sparse_csr()

    def _compute_dimensions(
        self,
        _scaled_height: Optional[int] = None,
        _scaled_width: Optional[int] = None,
    ):

        frame = self._get_frame(_probe=True)

        # NOTE: Dimension order is (H,W,D)
        # NOTE: Perhaps use transparency as well?
        original_shape = list(frame.shape)

        dim = DotMap()

        if len(frame.shape) == 2:
            original_shape.append(1)

        (dim.orig.H, dim.orig.W, dim.orig.D) = tuple(original_shape)
        (dim.H, dim.W, dim.D) = tuple(original_shape)
        dim.orig.span = dim.orig.H * dim.orig.W * dim.orig.D

        dim.resize = False

        if bool(_scaled_width) ^ bool(_scaled_height):
            if _scaled_height is not None:
                # Fixed height, calculate the width with the same AR
                pct = _scaled_height / float(original_shape[0])
                _scaled_width = int((float(original_shape[1]) * pct))

            elif _scaled_width is not None:
                # Fixed width, calculate the height with the same AR
                pct = _scaled_width / float(original_shape[1])
                _scaled_height = int((float(original_shape[0]) * pct))

            dim.H = _scaled_height
            dim.W = _scaled_width
            dim.span = dim.H * dim.W * dim.D
            dim.resize = True

        # Left and right padding
        left_right_padding = dim.W // 2 + dim.W % 2

        # Top and bottom padding
        top_bottom_padding = dim.H // 2 + dim.H % 2

        dim.padded.W = dim.W + 2 * left_right_padding
        dim.padded.H = dim.H * dim.D + 2 * top_bottom_padding
        dim.padded.D = dim.D
        dim.padded.span = dim.padded.W * dim.padded.H * dim.padded.D

        # The frame is padded only at the top and the bottom,
        # but the left and right padding values are used
        # to compute the size of the retinal field below.
        dim.padding = np.array(
            [
                left_right_padding,
                left_right_padding,
                top_bottom_padding,
                top_bottom_padding,
            ]
        )

        # print(f"==[ dim: {dim}")

        return dim

    def _make_rf(
        self,
        _k_max: int = 15,
        _k_min: int = 3,
        _sd_w: float = 1 / 4,
        _sd_h: float = 1 / 4,
    ):
        def gaussian_2d(
            _x: float,
            _y: float,
            _mx: float,
            _my: float,
            _sx: float,
            _sy: float,
        ):
            # Unnormalised 2D Gaussian.
            return pt.exp(
                -((_x - _mx) ** 2 / (2 * _sx ** 2) + (_y - _my) ** 2 / (2 * _sy ** 2))
            )

        # src = pt.ones(h, w)
        # plt.imshow(src)

        # shape = np.array(src.shape)
        # print(f"==[ shape: {shape}")

        # centre = shape // 2
        # print(f"==[ centre: {centre}")

        # max_side = max(shape)
        # print(f"==[ max_side: {max_side}")

        # Meshgrid of (x, y) coordinates for each pixel
        x = pt.linspace(0, self.dim.padded.W - 1, self.dim.padded.W)
        y = pt.linspace(0, self.dim.padded.H - 1, self.dim.padded.H)
        x, y = pt.meshgrid(x, y, indexing="ij")

        # Unnormalised 2D Gaussian representing the distribution
        # of kernel sizes away from the fovea.
        gaussian = gaussian_2d(
            x,
            y,
            _mx=self.dim.padded.W // 2,
            _my=self.dim.padded.H // 2,
            _sx=self.dim.W * _sd_w,
            _sy=self.dim.H * _sd_h,
        )

        # 　Multiply　by　the maximum kernel size
        ksizes = gaussian * _k_max

        # Stochastic blurring of the 'edges' created by
        # jumps in the kernel size
        blurred = pt.bernoulli(ksizes - pt.floor(ksizes))
        ksizes = pt.ceil(ksizes + blurred).int()
        # print(f'==[ blurred: {blurred}')

        # Set kernel sizes smaller than the minimum to the minimum
        # ksizes[ksizes < _k_min] = _k_min

        # Compute indices by kernel size
        indices = {}
        for i in range(max(ksizes.min(), _k_min), ksizes.max() + 1):
            indices[i] = pt.where(ksizes == i)

        rf_rows = []
        rf_cols = []
        rf_vals = []

        # printksize = _k_min + 1
        printksize = 0

        for ksize, (cols, rows) in indices.items():

            # if ksize < _k_min:
            #     continue

            diff = pt.linspace(-ksize // 2 + 1, ksize // 2, ksize)
            # print(f'==[ diff:\n{diff}')

            colspan = (
                ((cols[None, :].repeat(ksize, 1) + diff[:, None]))
                .t()
                .repeat_interleave(ksize, dim=1)
                .int()
            )
            rowspan = (
                (rows[None, :].repeat(ksize, 1) + diff[:, None])
                .t()
                .repeat(1, ksize)
                .int()
            )

            # if ksize == printksize:
            #     print(f"==[ colspan:\n{colspan}")
            #     print(f"==[ rowspan:\n{rowspan}")

            col_idx = colspan * self.dim.padded.H + rowspan

            row_idx = (
                pt.linspace(0, col_idx.shape[0] - 1, col_idx.shape[0])
                .int()[:, None]
                .repeat(1, ksize ** 2)
            )

            # Boundary check
            mask = (
                rowspan.ge(0)
                * rowspan.lt(self.dim.padded.H)
                * colspan.ge(0)
                * colspan.lt(self.dim.padded.W)
            ).flatten()

            if ksize == printksize:
                print(f"==[ mask: {mask}")

            col_idx = col_idx.flatten()
            row_idx = row_idx.flatten()
            # print(f'==[ col_idx:\n{col_idx}')
            # print(f'==[ row_idx:\n{row_idx}')

            sp_rows = (
                (cols * self.dim.padded.H + rows)
                .int()[:, None]
                .repeat(1, ksize ** 2)
                .flatten()
            )
            # print(f"==[ kernel {ksize} ] sp_rows:\n{sp_rows}")

            combined = pt.cat((col_idx[:, None], sp_rows[:, None]), dim=1)

            masked = combined[mask]
            if ksize == printksize:
                print(f"==[ combined:\n{combined}")
                print(f"==[ masked:\n{masked}")
            # print(f"==[ masked shape: {masked.shape}")

            rf_cols.extend(masked[:, 0].tolist())
            rf_rows.extend(masked[:, 1].tolist())
            rf_vals.extend([1 / ksize ** 2] * masked.shape[0])

        # print(f"==[ rf_cols length: {len(rf_cols)}")
        # print(f"==[ rf_rows length: {len(rf_rows)}")
        # print(f"==[ rf_vals length: {len(rf_vals)}")

        rf = (
            pt.sparse_coo_tensor(
                np.array(
                    [
                        rf_rows,
                        rf_cols,
                    ]
                ),
                np.array(rf_vals),
                size=(
                    self.dim.padded.span,
                    self.dim.padded.span,
                ),
                dtype=pt.float32,
            )
            .coalesce()
            .to_sparse_csr()
        )

        # print(f"==[ rf size: {rf.size()}")
        # print(f"==[ rf nnz: {rf._nnz()}")

        # print(f"==[ mask: {mask.t()}")

        return (
            rf,
            gaussian.t(),
        )

    def _normalise(
        self,
        _frame: pt.Tensor,
        _min: Optional[float] = 0.0,
        _max: Optional[float] = 255.0,
    ) -> pt.Tensor:
        """
        Min-max normalised version of the frame.
        """

        fmin = float(pt.min(_frame))
        fmax = float(pt.max(_frame))

        return _min + (_max - _min) * (_frame - fmin) / (fmax - fmin)

    def _stretch(
        self,
        _frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Stretch a 2D image into a 1D vector.
        """

        if self.dim.D > 1:
            # Transpose the depth dimension and stretch
            return _frame.transpose(2, 1, 0).flatten()

        return _frame.t().flatten()[:, None]

    def _fold(
        self,
        _frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Fold a 1D vector into a 2D image.
        """

        # print(f"==[ frame shape: {_frame.shape}")

        return _frame.reshape(
            self.dim.padded.W,
            self.dim.padded.H,
        ).t()

    def _get_padding(
        self,
        _width_offset: float = 0.0,
        _height_offset: float = 0.0,
    ) -> pt.Tensor:

        """
        Compute the horizontal and vertical offsets
        from the width and height offset values and
        then compute the padding values from the offsets.
        """

        # TODO boundary check

        horizontal_offset_pixels = int(
            np.sign(_width_offset) * math.floor(math.fabs(_width_offset) * self.dim.W)
        )
        vertical_offset_pixels = int(
            np.sign(_height_offset) * math.floor(math.fabs(_height_offset) * self.dim.H)
        )

        padding = self.dim.padding + np.array(
            [
                horizontal_offset_pixels,
                -horizontal_offset_pixels,
                vertical_offset_pixels,
                -vertical_offset_pixels,
            ]
        )

        # print(f"==[ padding: {padding}")

        return tuple(padding.tolist())

    def _local_mean(
        self,
        _frame: pt.Tensor,
        _padding: Optional[Tuple[int, int, int, int]] = None,
    ) -> pt.Tensor:

        # Sanity check for the padding
        if _padding is None:
            _padding = tuple()

        # Pad the frame so that we can shift the FOV
        # without making the frame 'jump'
        padded = ptf.pad(_frame, _padding)
        stretched = self._stretch(padded)
        convolved = pt.mm(self.rf, stretched)

        return convolved

    # *  Public methods  * #

    def read(
        self,
        _width_offset: float = 0.0,
        _height_offset: float = 0.0,
        _views: Optional[List[View]] = None,
    ) -> pt.Tensor:
        """
        Read the input by applying a certain offset:
        - Read the next frame
            - (Optional) flatten the frame (remove all channel information)
        - Get the input padding corresponding to the specified offset
        - Compute the local contrast normalisation
        """

        views = []

        if _views is None:
            _views = {View.Original}

        frame = self._get_frame()

        if View.Original in _views:
            views.append(frame)

        # Flatten the frame
        if self.flatmask is not None:
            frame *= self.flatmask

        padding = self._get_padding(_width_offset, _height_offset)

        local_mean = self._local_mean(frame, padding)

        local_mean = self._fold(local_mean)[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

        local_mean *= self.norm_mask[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

        # print(f"==[ local_mean: {local_mean}")

        # Subtract the mean and normalise
        norm = self._normalise(frame - local_mean)

        if View.Normalised in _views:
            views.append(norm)

        if View.LocalMean in _views:
            views.append(local_mean)

        return views

    def show(
        self,
        _views: List[pt.Tensor],
    ):

        if len(_views) == 0:
            return

        # Show all images
        # for idx, _img in enumerate(_frames):
        cv.imshow(
            f"Result",
            np.hstack([view.numpy() for view in _views]).astype(np.uint8),
        )

        # Press ESC to quit
        self.processing &= cv.waitKey(10) != 27
