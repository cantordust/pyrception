from __future__ import annotations

# --------------------------------------
from typing import *

# --------------------------------------
from pathlib import Path

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from loguru import logger

# --------------------------------------
from skimage import draw

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
from dotmap import DotMap

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf

pt.set_printoptions(linewidth=200)

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import KernelType

# --------------------------------------
from functools import partial


class ProtoLayer:

    """
    A proto-layer serving as a base class to all the other retinal layers.
    """

    def __init__(
        self,
        h: int,
        w: int,
        sectors: int = 36,
        kernel_type: KernelType = KernelType.Proportional,
        kernel_scale: float = 1.0,
        kernel_params: Dict[str, Any] = None,
        extent: int = None,
        cutoff: float = 1e-3,
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
    ):
        if kernel_params is None:
            kernel_params = {}

        self.h = h
        self.w = w
        self.sectors = sectors
        self.kernel_type = kernel_type
        self.kernel_scale = kernel_scale
        self.kernel_params = kernel_params
        self.extent = extent
        self.cutoff = cutoff

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(scaled_height, scaled_width, saccades)

        # Kernel factory
        # ==================================================
        self.kernel_function = None

        if kernel_type == KernelType.Proportional:
            self.kernel_function = self._make_proportional_kernels

        elif kernel_type == KernelType.Gaussian:
            self.kernel_function = self._make_gaussian_kernels

        elif kernel_type == KernelType.Gabor:
            self.kernel_function = self._make_gabor_kernels

        # Internal parameters used for constructing kernels
        # ==================================================
        self.Xs = None
        self.Ys = None

        self.rows = None
        self.cols = None
        self.vals = None
        self.coords = None
        self.rfs = None

    def _make_mesh(self) -> pt.Tensor:
        if self.Xs is not None:
            return

        cx = pt.linspace(0, self.h - 1, self.h)
        cy = pt.linspace(0, self.w - 1, self.w)

        (self.Xs, self.Ys) = pt.meshgrid(cy, cx, indexing="ij")

    def _trim(
        self,
        indices: pt.Tensor,
    ):
        # Mask to trim the indices to the image dimensions
        mask = (
            (indices[:, 0] < self.w)
            & (indices[:, 0] >= 0)
            & (indices[:, 1] < self.h)
            & (indices[:, 1] >= 0)
        )

        # Keep only unique pairs of indices
        indices = pt.unique(indices[mask], dim=0)

        return indices

    def _compute_dimensions(
        self,
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
    ):
        dim = DotMap()
        original_shape = [self.h, self.w, 1]

        (dim.orig.H, dim.orig.W, dim.orig.D) = tuple(original_shape)
        (dim.H, dim.W, dim.D) = tuple(original_shape)
        dim.orig.span = dim.orig.H * dim.orig.W * dim.orig.D

        dim.resize = False

        if bool(scaled_width) ^ bool(scaled_height):
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
            dim.span = dim.H * dim.W * dim.D
            dim.resize = True

        # Left and right padding
        lr_padding = (dim.W // 2 + dim.W % 2) if saccades else 0

        # Top and bottom padding
        tb_padding = (dim.H // 2 + dim.H % 2) if saccades else 0

        dim.padded.W = dim.W + 2 * lr_padding
        dim.padded.H = dim.H * dim.D + 2 * tb_padding
        dim.padded.D = dim.D
        dim.padded.span = dim.padded.W * dim.padded.H * dim.padded.D

        # The frame is padded only at the top and the bottom,
        # but the left and right padding values are used
        # to compute the size of the retinal field below.
        dim.padding = np.array(
            [
                lr_padding,
                lr_padding,
                tb_padding,
                tb_padding,
            ]
        )

        return dim

    def _make_proportional_kernels(
        self,
        mx: int,
        my: int,
        radius: float,
        scale: float = 1,
        fill: float = None,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Flat kernel with weights proportional to the RF size.

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        radius *= scale

        # Get the coordinates of a disk with the given
        # centre coordinates and radius
        (rows, cols) = draw.disk((mx, my), max(radius, 1))
        k_idx = pt.LongTensor(list(zip(rows, cols)))

        # Cut off coordinates that do not fit into the image
        coords = self._trim(k_idx)
        (cols, rows) = coords[:, 0], coords[:, 1]

        if rows.numel() > 0:
            if fill is None:
                # fill = radius**-2
                fill = 1 / (1 + rows.numel())
            values = [fill] * rows.shape[0]

            # Return the kernel indices and values
            return (rows, cols, values)

    def _make_gaussian_kernels(
        self,
        mx: int,
        my: int,
        sd: float,
        scale: float = 1,
        norm: bool = True,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        2D Gaussian kernel with given mean and SD.

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        # Gaussian distribution
        sd *= scale

        values = pt.exp(
            -0.5 * (((self.Xs - mx) / sd) ** 2 + ((self.Ys - my) / sd) ** 2)
        )

        # Limit the kernel to values above the cutoff
        k_idx = pt.argwhere(values >= self.cutoff)

        # Cut off coordinates that do not fit into the image
        coords = self._trim(k_idx)
        (cols, rows) = coords[:, 0], coords[:, 1]

        # Normalise if necessary
        if norm:
            values /= 2 * pt.pi * sd**2

        if rows.numel() > 0:
            values = values[cols, rows]

            # Return the kernel indices and values
            return (rows, cols, values.flatten())

    def _make_gabor_kernels(
        self,
        mx: int,
        my: int,
        sd: float,
        orientation: float = 0.0,  # Orientation [deg]
        frequency: float = 0.1,  # Sine component frequency
        aspect: float = 0.1,  # Aspect ratio
        phase: float = 0.0,  # Phase of the sine component
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        WIP - do not use!

        Args:
            mx (int): _description_
            my (int): _description_
            sd (float): _description_
            orientation (float, optional): _description_. Defaults to 0.0.
            frequency (float, optional): _description_. Defaults to 0.1
            aspect (float, optional): _description_. Defaults to 0.1
            phase (float, optional): _description_. Defaults to 0.0

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        return pt.from_numpy(
            cv.getGaborKernel((mx, my), sd, orientation, frequency, aspect, phase)
        )

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

    def make_rfs(
        self,
        inverse: bool = False,
    ) -> pt.Tensor:
        """
        Implementation of eccentricity-dependent logpolar distribution of receptive fields as presented in

        Maiello, G., Chessa, M., Bex, P. J. & Solari, F. Near-optimal combination of disparity across a log-polar scaled visual field. PLoS Comput Biol 16, e1007699 (2020).

        Args:

        Returns:
            pt.Tensor:
                The receptive fields.
        """

        # Prepare the mesh
        self._make_mesh()

        # Construct the receptive fields
        # ==================================================

        # Coordinates of the central pixel
        w2 = self.w // 2
        h2 = self.h // 2

        # Maximal radius of the centre of the log-polar rings
        rho_max = math.sqrt(h2**2 + w2**2)

        # Size of the foveal region
        rho_f = math.log(rho_max)

        # Number of sectors (ideally should be divisible by 4)
        S = self.sectors

        # Sector size
        q = S / (2 * pt.pi)

        # Growth factor for coupling S with R below.
        # This preserves the pixel aspect ratio.
        a = 1 + 1 / q

        # Number of radial rings (coupled with the number of sectors)
        # R = np.floor(1 / np.emath.logn(rho_max / rho_f, a)).astype(np.int32)
        log_a = math.log(a)
        R = math.floor(math.log(rho_max / rho_f) / log_a)

        # Cartesian coordinate mesh (x, y)
        rf_xs = pt.linspace(-h2, h2 + 1, self.h + 1)
        rf_ys = pt.linspace(-w2, w2 + 1, self.w + 1)
        Xs, Ys = pt.meshgrid(rf_xs, rf_ys, indexing="ij")

        # Logpolar coordinate mesh (r, φ)
        radii = pt.sqrt(Xs**2 + Ys**2)
        ratios = Xs / radii
        angles = pt.arccos(ratios)
        angles = pt.where(Ys > 0, 2 * pt.pi - angles, angles)

        # Eccentricity-dependent logpolar coordinates
        ksi = pt.log(radii / rho_f) / log_a
        eta = q * angles

        # Discrete versions of ksi and eta (mappable to pixel coordinates)
        u = pt.floor(ksi)
        v = pt.floor(eta)

        # Mesh of discrete logpolar coordinates of the RF centres
        # Rs: radii
        # Ts: angles
        rpoints = pt.unique(rho_f * (a**u))
        tpoints = pt.unique(v / q)
        (Rs, Ts) = pt.meshgrid(rpoints, tpoints, indexing="ij")

        # Convert Rs and Ts back to integer (x, y) coordinates
        rf_xs = pt.round(pt.cos(Ts) * Rs) + w2
        rf_ys = pt.round(pt.sin(Ts) * Rs) + h2

        # print(f"==[ {rf_xs.shape}")
        # print(f"==[ {rf_ys.shape}")

        # Cut off coordinates that do not fit into the image
        coords = pt.unique(
            pt.stack((rf_xs.flatten(), rf_ys.flatten()), dim=1), dim=0
        ).int()
        coords = self._trim(coords)

        # Maximal RF size
        if inverse:
            # TODO This is ad hoc, needs to be adjusted
            W_init = (rho_f / rho_max) * (a**R) * (1 - 1 / a)
        else:
            W_init = (rho_f / rho_max) * (a**R) * (1 - 1 / a)

        print(f"==[ W_max: {W_init}")

        # Finally, compute the kernels.
        # ==================================================
        rows = []
        cols = []
        values = []
        indices = []
        rf_coords = []

        extent = rho_max if self.extent is None else self.extent * rho_max

        for idx, (rx, ry) in enumerate(coords):
            cdist = pt.sqrt((rx - w2) ** 2 + (ry - h2) ** 2)
            # print(f"==[ {rx, ry}")

            if self.extent is not None and cdist > extent:
                continue

            if inverse:
                radius = max(1, self.kernel_scale * W_init * (rho_max - cdist + 1))
                if radius < 3:
                    continue
            else:
                radius = self.kernel_scale * W_init * cdist

            result = self.kernel_function(
                rx,
                ry,
                max(1, radius),
                **self.kernel_params,
            )

            if result is not None:
                (rs, cs, vs) = result

                rows.append(rs.tolist())
                cols.append(cs.tolist())
                values.append(vs)
                indices.append([len(indices)] * len(rs))
                rx = max(0, min(rx, self.w - 1))
                ry = max(0, min(ry, self.h - 1))
                rf_coords.append([ry, rx])
                # print(f"==[ rs: {rs.shape}")
                # print(f"==[ {idx} ] indices: {len(indices[-1])}")

        # Prepare the sparse tensor coordinates and values
        sp_rows = np.concatenate(indices)
        sp_cols = np.concatenate(rows) + self.h * np.concatenate(cols)
        sp_vals = np.concatenate(values)

        sp_indices = np.vstack(
            [
                sp_rows,
                sp_cols,
            ]
        )

        rfs = (
            pt.sparse_coo_tensor(
                sp_indices,
                sp_vals,
                size=(
                    len(indices),
                    self.dim.padded.span,
                ),
                dtype=pt.float32,
            )
            .coalesce()
            .to_sparse_csr()
        )

        self.rows = rows
        self.cols = cols
        self.vals = values
        self.coords = np.array(rf_coords)
        self.rfs = rfs

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

    def __del__(self):
        # TODO Add code for releasing individual
        # VideoWriter sinks for different views.
        pass

    def _as_image(
        self,
        frame: np.ndarray,
    ):
        fmin = frame.min()

        return 255 * (frame - fmin) / (frame.max() - fmin + 1e-8)

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

    # def _compute_dimensions(
    #     self,
    #     original_shape: List[int],
    #     scaled_height: Optional[int] = None,
    #     scaled_width: Optional[int] = None,
    #     saccades: bool = False,
    # ):
    #     dim = DotMap()

    #     if len(original_shape) == 2:
    #         original_shape.append(1)

    #     # NOTE: Dimension order is (H,W,D)
    #     # NOTE: Perhaps use transparency as well?
    #     dim.orig.shape = original_shape
    #     (dim.orig.H, dim.orig.W, dim.orig.D) = tuple(original_shape)
    #     dim.orig.span = dim.orig.H * dim.orig.W * dim.orig.D
    #     (dim.H, dim.W, dim.D) = tuple(original_shape)
    #     dim.shape = original_shape

    #     dim.resize = False

    #     if bool(scaled_height) ^ bool(scaled_width):
    #         if scaled_height is not None:
    #             # Fixed height, calculate the width with the same AR
    #             pct = scaled_height / float(original_shape[0])
    #             scaled_width = int((float(original_shape[1]) * pct))

    #         elif scaled_width is not None:
    #             # Fixed width, calculate the height with the same AR
    #             pct = scaled_width / float(original_shape[1])
    #             scaled_height = int((float(original_shape[0]) * pct))

    #         dim.H = scaled_height
    #         dim.W = scaled_width
    #         dim.shape = (dim.H, dim.W, dim.D)
    #         dim.span = dim.H * dim.W * dim.D
    #         dim.resize = True

    #     # Left and right padding
    #     left_right_padding = dim.W // 2 + dim.W % 2 if saccades else 0

    #     # Top and bottom padding
    #     top_bottom_padding = dim.H // 2 + dim.H % 2 if saccades else 0

    #     dim.padded.W = dim.W + 2 * left_right_padding
    #     dim.padded.H = dim.H * dim.D + 2 * top_bottom_padding
    #     dim.padded.D = dim.D
    #     dim.padded.span = dim.padded.W * dim.padded.H * dim.padded.D

    #     # The frame is padded only at the top and the bottom,
    #     # but the left and right padding values are used
    #     # to compute the size of the retinal field below.
    #     dim.padding = np.array(
    #         [
    #             left_right_padding,
    #             left_right_padding,
    #             top_bottom_padding,
    #             top_bottom_padding,
    #         ]
    #     )

    #     # print(f"==[ dim: {dim}")

    #     return dim

    def make_rf_phyllotactic(
        h: int = 256,
        w: int = 256,
        r0: float = 0.1,
        z: float = 1.0005,
        div: float = 2 / (1 + np.sqrt(5)),
        sd_coeff: float = 0.4,
    ):
        """
        WIP - do not use!

        A version of the logpolar distribution that follows a phyllotactic pattern.
        This type of pattern is observed in the receptor layer.

        Returns:
            pt.Tensor:
                Kernels (and the kernel indices) arranged in a phyllotactic pattern.
        """
        kernels = {}
        mw = w // 2
        mh = h // 2
        x = mh
        y = mw

        diag = np.sqrt((h - mh) ** 2 + (w - mw) ** 2)

        n = 0
        while np.sqrt((x - mh) ** 2 + (y - mw) ** 2) <= diag:
            r_n = r0 * z**n
            theta_n = 2 * np.pi * n * div

            x = int(r_n * (z**theta_n) * np.sin(theta_n)) + mh
            y = int(r_n * (z**theta_n) * np.cos(theta_n)) + mw
            if 0 <= x < h and 0 <= y < w:
                if (x, y) not in kernels:
                    sd = sd_coeff * r0 * np.sqrt((x - mh) ** 2 + (y - mw) ** 2)
                    kernel = np.nan_to_num(
                        ProtoLayer._make_gaussian_kernels(h, w, x, y, sd, sd).numpy(),
                        nan=0.0,
                    )
                    kernels[(x, y)] = kernel

            n += 1

        return kernels

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
        rftype: KernelType = KernelType.Proportional,
        scale: float = 1.0,
        norm: bool = False,
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

            # Row and column spans for the current kernel size
            # ==================================================
            kspan = pt.linspace(-ksize // 2 + 1, ksize // 2, ksize)
            # print(f'==[ diff:\n{diff}')

            # Rows spanned by stretched kernels
            # ==================================================
            krowspan = (
                (rows[None, :].repeat(ksize, 1) + kspan[:, None])
                .t()
                .repeat(1, ksize)
                .int()
            ).flatten()

            # Columns spanned by stretched kernels
            # ==================================================
            kcolspan = (
                (cols[None, :].repeat(ksize, 1) + kspan[:, None])
                .t()
                .repeat_interleave(ksize, dim=1)
                .int()
            ).flatten()

            # Boundary check mask
            # ==================================================
            boundary_mask = (
                krowspan.ge(0)
                * krowspan.lt(self.dim.padded.H)
                * kcolspan.ge(0)
                * kcolspan.lt(self.dim.padded.W)
            ).flatten()

            # Sparse row and column indices
            # ==================================================
            sp_rows = (
                (cols * self.dim.padded.H + rows)
                .int()[:, None]
                .repeat(1, ksize**2)
                .flatten()
            )
            sp_cols = kcolspan * self.dim.padded.H + krowspan

            # Combined sparse indices with boundary check
            # ==================================================
            sp_indices = pt.cat((sp_rows[:, None], sp_cols[:, None]), dim=1)[
                boundary_mask
            ]

            sp_rows = sp_indices[:, 0]
            sp_cols = sp_indices[:, 1]

            rf_rows.extend(sp_rows.tolist())
            rf_cols.extend(sp_cols.tolist())

            if rftype == KernelType.Proportional:
                (rows, cols, kernel) = self._make_gaussian_kernels(
                    ksize,
                    scale,
                    norm,
                    rows,
                    boundary_mask,
                )

            elif rftype == KernelType.Gabor:
                kvals = self._make_gaussian_kernels(
                    rftype,
                    boundary_mask=boundary_mask,
                )

            elif rftype == KernelType.Proportional:
                (rows, cols, kernel) = self._make_proportional_kernels(
                    ksize,
                    scale,
                    sp_rows,
                )

            else:
                raise ValueError(f"Invalid RF type '{rftype}'")

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
