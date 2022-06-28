# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from __future__ import annotations

# --------------------------------------
from typing import Set
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
from matplotlib import pyplot as plt

# --------------------------------------
from dotmap import DotMap

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelSizeDist
from pyrception.visual.util.types import KernelWeightDist


class ProtoLayer:

    """
    A proto-layer serving as a base class to all the other retinal classes.
    """

    @staticmethod
    def _make_mesh(
        h: int,
        w: int,
    ):

        x = pt.linspace(0, h - 1, h)
        y = pt.linspace(0, w - 1, w)

        x, y = pt.meshgrid(x, y, indexing="ij")

        return x, y

    @staticmethod
    def _kdist_gaussian(
        h: int,
        w: int,
        mh: int,
        mw: int,
        sh: int = 1 / 16,
        sw: int = 1 / 16,
        decreasing: bool = True,
        norm: bool = False,
    ):
        x, y = ProtoLayer._make_mesh(h, w)

        # NOTE: this is an *unnormalised* gaussian
        dist = pt.exp(
            -((x - mh) ** 2 / (2 * (sh * h) ** 2) + (y - mw) ** 2 / (2 * (sw * w) ** 2))
        )

        if norm:
            dist /= 2 * pt.pi * sh * h * sw * w

        return dist if decreasing else 1 - dist

    @staticmethod
    def _kdist_logpolar(
        h: int,
        w: int,
        sh: int = 1 / 16,
        sw: int = 1 / 16,
        decreasing: bool = True,
    ):
        x, y = ProtoLayer._make_mesh(h, w)
        dist = pt.log1p(1.0 + pt.sqrt(sh * (x - h // 2) ** 2 + sw * (y - w // 2) ** 2))

        return 1 / (1 + dist) if decreasing else dist

    @staticmethod
    def _kdist_flat(
        h: int,
        w: int,
    ):

        return pt.ones(h, w).int()

    @staticmethod
    def scale(
        tensor: pt.Tensor,
        min: Optional[float] = 0.0,
        max: Optional[float] = 255.0,
    ) -> pt.Tensor:
        """
        Min-max normalised version of the frame.
        """

        tmin = tensor.min()
        tmax = tensor.max()

        return min + (max - min) * (tensor - tmin) / (tmax - tmin)

    @staticmethod
    def stretch(frame: pt.Tensor) -> pt.Tensor:
        """
        Stretch a 2D image into a 1D vector.
        """

        # TODO: Handle transparency (4D tensors)?
        if frame.dim() == 3:
            # Transpose the depth dimension and stretch
            return frame.permute(2, 1, 0).flatten()[:, None]

        return frame.permute(1, 0).flatten()[:, None]

    @staticmethod
    def fold(
        frame: pt.Tensor,
        h: int,
        w: int,
    ) -> pt.Tensor:
        """
        Fold a 1D vector into a 2D tensor.
        """

        # print(f"==[ frame shape: {frame.shape}")

        return frame.reshape(w, h).t()

    @staticmethod
    def get_kdist(
        h: int,
        w: int,
        k_min: Optional[int] = 3,
        k_max: Optional[int] = 15,
        mh: Optional[int] = None,
        mw: Optional[int] = None,
        sh: Optional[int] = 1 / 16,
        sw: Optional[int] = 1 / 16,
        kdist: KernelSizeDist = KernelSizeDist.Gaussian,
        decreasing: bool = True,
        smooth: bool = True,
    ):

        if mh is None:
            mh = h // 2

        if mw is None:
            mw = w // 2

        if kdist == KernelSizeDist.Flat:
            # * Flat kernel distribution * #
            kdist = ProtoLayer._kdist_flat(h, w)
            ksizes = kdist * k_min

        else:
            if kdist == KernelSizeDist.LogPolar:
                # * Log-polar kernel distribution * #
                kdist = ProtoLayer._kdist_logpolar(h, w, sh, sw, decreasing)

            elif kdist == KernelSizeDist.Gaussian:
                # * Gaussian kernel distribution * #
                kdist = ProtoLayer._kdist_gaussian(h, w, mh, mw, sh, sw, decreasing)

            ksizes = ProtoLayer.scale(kdist, k_min, k_max)

            # Stochastic blurring of the 'edges'
            # created by jumps in the kernel size.
            if smooth:
                smoothing = pt.bernoulli(ksizes - pt.floor(ksizes))
                ksizes = pt.ceil(ksizes + smoothing).int()

            # Scaling between k_min and k_max
            kdist = ProtoLayer.scale(kdist, 0.0, 1.0)

        return (kdist, ksizes)

    @staticmethod
    def pad(
        frame: pt.Tensor,
        padding: Tuple[int, int, int, int],
    ) -> pt.Tensor:

        # Pad the frame so that we can shift the FOV
        # without making the frame 'jump'
        padded = ptf.pad(frame, padding)

        return padded

    @staticmethod
    def unpad(
        frame: pt.Tensor,
        padding: Tuple[int, int, int, int],
    ):

        return frame[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

    def _get_padding(
        self,
        height_offset: float = 0.0,
        width_offset: float = 0.0,
    ) -> pt.Tensor:

        """
        Compute the horizontal and vertical offsets
        from the width and height offset values and
        then compute the padding values from the offsets.
        """

        # TODO boundary checks

        # wop: width offset pixels
        # hop: height offset pixels
        wop = int(
            np.sign(width_offset) * math.floor(math.fabs(width_offset) * self.dim.W)
        )
        hop = int(
            np.sign(height_offset) * math.floor(math.fabs(height_offset) * self.dim.H)
        )

        padding = tuple(
            self.dim.padding
            + np.array(
                [
                    hop,
                    -hop,
                    wop,
                    -wop,
                ]
            ).tolist()
        )

        return padding

    def _compute_dimensions(
        self,
        original_shape: List[int],
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
    ):
        dim = DotMap()

        if len(original_shape) == 2:
            original_shape.append(1)

        # NOTE: Dimension order is (H,W,D)
        # NOTE: Perhaps use transparency as well?
        dim.orig.shape = original_shape
        (dim.orig.H, dim.orig.W, dim.orig.D) = tuple(original_shape)
        dim.orig.span = dim.orig.H * dim.orig.W * dim.orig.D
        (dim.H, dim.W, dim.D) = tuple(original_shape)
        dim.shape = original_shape

        dim.resize = False

        if bool(scaled_height) ^ bool(scaled_width):
            if scaled_height is not None:
                # Fixed height, calculate the width with the same AR
                pct = scaled_height / float(original_shape[0])
                scaled_width = int((float(original_shape[1]) * pct))

            elif scaled_width is not None:
                # Fixed width, calculate the height with the same AR
                pct = scaled_width / float(original_shape[1])
                scaled_height = int((float(original_shape[0]) * pct))

            dim.H = scaled_height
            dim.W = scaled_width
            dim.shape = (dim.H, dim.W, dim.D)
            dim.span = dim.H * dim.W * dim.D
            dim.resize = True

        # Left and right padding
        left_right_padding = dim.W // 2 + dim.W % 2 if saccades else 0

        # Top and bottom padding
        top_bottom_padding = dim.H // 2 + dim.H % 2 if saccades else 0

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

    def _convolve(
        self,
        frame: pt.Tensor,
        rf: pt.Tensor,
        height: int,
        width: int,
    ) -> pt.Tensor:
        """
        Convolve the input frame with the current layer's receptive field.

        The height and width parameters are necessary to fold the convolved stretched frame
        back into a 2D frame with the right dimensions.

        Returns:

        Args:
            frame (pt.Tensor):
                The input frame.

            rf (pt.Tensor):
                Receptive field

            height (int):
                The height of the folded convolved image.

            width (int):
                The width of the folded convolved image.

        Returns:
            pt.Tensor:
                A map of the local mean illumination at each pixel.
        """

        stretched = self.stretch(frame)

        mean = pt.mm(rf, stretched)

        folded = self.fold(mean, height, width)

        return folded

    def _make_rf(
        self,
        ksizes: pt.Tensor,
        kwdist: KernelWeightDist = KernelWeightDist.Proportional,
        scale: float = 1.0,
    ):

        # Compute tensor indices for each kernel size
        indices = {}
        for i in range(ksizes.min(), ksizes.max() + 1):
            indices[i] = pt.where(ksizes == i)

        rf_rows = []
        rf_cols = []
        rf_vals = []

        for ksize, (rows, cols) in indices.items():

            if rows.numel() == 0:
                continue

            # * Row and column spans for the current kernel size * #
            kspan = pt.linspace(-ksize // 2 + 1, ksize // 2, ksize)
            # print(f'==[ diff:\n{diff}')

            # Rows spanned by stretched kernels
            krowspan = (
                (rows[None, :].repeat(ksize, 1) + kspan[:, None])
                .t()
                .repeat(1, ksize)
                .int()
            ).flatten()

            # Columns spanned by stretched kernels
            kcolspan = (
                (cols[None, :].repeat(ksize, 1) + kspan[:, None])
                .t()
                .repeat_interleave(ksize, dim=1)
                .int()
            ).flatten()

            # * Boundary check mask * #
            boundary_mask = (
                krowspan.ge(0)
                * krowspan.lt(self.dim.padded.H)
                * kcolspan.ge(0)
                * kcolspan.lt(self.dim.padded.W)
            ).flatten()

            # * Sparse row and column indices * #
            sp_rows = (
                (cols * self.dim.padded.H + rows)
                .int()[:, None]
                .repeat(1, ksize**2)
                .flatten()
            )
            sp_cols = kcolspan * self.dim.padded.H + krowspan

            # * Combined sparse indices with boundary check * #
            sp_indices = pt.cat((sp_rows[:, None], sp_cols[:, None]), dim=1)[
                boundary_mask
            ]

            sp_rows = sp_indices[:, 0]
            sp_cols = sp_indices[:, 1]

            rf_rows.extend(sp_rows.tolist())
            rf_cols.extend(sp_cols.tolist())

            if kwdist == KernelWeightDist.Gaussian:

                mean = ksize // 2

                if ksize % 2 == 0:
                    mean -= 0.5

                kvals = (
                    ProtoLayer._kdist_gaussian(
                        ksize,
                        ksize,
                        mean,
                        mean,
                        0.5,
                        0.5,
                        norm=False,
                    )
                    .repeat(len(rows), 1)
                    .flatten()[boundary_mask]
                    * scale
                ).tolist()

            elif kwdist == KernelWeightDist.Proportional:
                val = (1 / ksize**2) * scale
                kvals = [val] * sp_rows.shape[0]

            else:
                kvals = [scale] * sp_rows.shape[0]

            rf_vals.extend(kvals)

        rf = pt.sparse_coo_tensor(
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

        return rf.coalesce().to_sparse_csr()
