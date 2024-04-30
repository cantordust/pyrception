from typing import *

# --------------------------------------
from skimage import draw

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception.conf import logger
from pyrception import conf
from pyrception.visual.util.types import KernelType
from pyrception.visual.layers.base import BaseLayer


class ProtoLayer(BaseLayer):
    """
    A proto-layer serving as a base class to all the other retinal layers.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        sectors: int = 32,
        kernel_type: KernelType = KernelType.Flat,
        kernel_scale: float = 1.0,
        kernel_params: Dict[str, Any] = None,
        min_kernel_size: int = 1,
        extent: int = 1.0,
        fovea_ratio: int = 1.0,
        kernel_cutoff: float = 1e-3,
        inverse: bool = False,
        dense: bool = False,
        name: str = "Proto",
    ):

        name = f"{name:<10s}"
        super().__init__(name)

        self.info("Initialising...")

        if kernel_params is None:
            kernel_params = {}

        # TODO: add some assertions
        self.h = shape[0]
        self.w = shape[1]
        self.sectors = sectors
        self.kernel_type = kernel_type
        self.kernel_scale = kernel_scale
        self.kernel_params = kernel_params
        self.min_kernel_size = min_kernel_size
        self.extent = extent
        self.fovea_ratio = fovea_ratio
        self.kernel_cutoff = kernel_cutoff

        self.dims = self._compute_dimensions((self.h, self.w))

        # Kernel factory
        # ==================================================
        self.kernel_function = self._get_kernel_function(kernel_type)

        # Internal parameters used for constructing kernels
        # ==================================================
        self.Xs = None
        self.Ys = None

        self.rows = None
        self.cols = None
        self.vals = None
        self.cell_coords = None
        self.rfs = None
        self.neuron_count = 0
        self.rf_factors = None

        self._make_eccentric_rfs(inverse, dense)

    def _get_kernel_function(
        self,
        kernel_type: KernelType = KernelType.Flat,
    ):

        kernel_functions = {
            KernelType.Flat: self._make_flat_kernels,
            KernelType.Gaussian: self._make_gaussian_kernels,
            KernelType.DoG: self._make_dog_kernels,
            KernelType.Gabor: self._make_gabor_kernels,
        }

        kernel_function = kernel_functions.get(kernel_type, None)

        if kernel_function is None:
            raise TypeError(f"Invalid kernel type '{kernel_type}'")

        return kernel_function

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

    def _make_flat_kernels(
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
                # This effectively takes the *mean* of the input
                # within the receptive field.
                # Useful for horizontal cells.
                fill = 1 / rows.numel()
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
        k_idx = pt.argwhere(values >= self.kernel_cutoff)

        # Cut off coordinates that do not fit into the image
        coords = self._trim(k_idx)
        (cols, rows) = coords[:, 0], coords[:, 1]

        # Normalise if necessary
        if norm:
            values /= 2 * pt.pi * sd**2

        if rows.numel() > 0:
            values = values[cols, rows].flatten()

            # Limit to reasonable values that are not too small
            limit = 0.01 * values.max()
            limit_coords = values >= limit
            values = values[limit_coords]
            rows = rows[limit_coords]
            cols = cols[limit_coords]

            # Return the kernel indices and values
            return (rows, cols, values)

    def _make_dog_kernels(
        self,
        mx: int,
        my: int,
        sd: float,
        scale: float = 1,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        2D difference-of-Gaussians (DoG) kernels with given mean and SD.

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        # Gaussian distribution
        sd *= scale

        sd_narrow = sd / 2
        sd_wide = sd

        values_narrow = pt.exp(
            -0.5 * (((self.Xs - mx) / sd_narrow) ** 2 + ((self.Ys - my) / sd_narrow) ** 2)
        )
        values_wide = pt.exp(
            -0.5 * (((self.Xs - mx) / sd_wide) ** 2 + ((self.Ys - my) / sd_wide) ** 2)
        )

        # Limit the kernel to values where the *wide* Gaussian is above the cutoff
        k_idx = pt.argwhere(values_wide >= self.kernel_cutoff)

        # Cut off coordinates that do not fit into the image
        coords = self._trim(k_idx)
        (cols, rows) = coords[:, 0], coords[:, 1]

        # Normalise
        values_narrow /= 2 * pt.pi * sd_narrow**2
        values_wide /= 2 * pt.pi * sd_wide**2

        # DoG
        values = values_narrow - values_wide

        if rows.numel() > 0:
            values = values[cols, rows].flatten()
            values_wide = values_wide[cols, rows].flatten()

            # Limit to reasonable values that are not too small
            limit = 0.01 * values_wide.max()
            limit_coords = values_wide >= limit
            values = values[limit_coords]
            rows = rows[limit_coords]
            cols = cols[limit_coords]

            # Return the kernel indices and values
            return (rows, cols, values)

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
            mx (int):
                _description_

            my (int):
                _description_

            sd (float):
                _description_

            orientation (float, optional):
                _description_. Defaults to 0.0.

            frequency (float, optional):
                _description_. Defaults to 0.1

            aspect (float, optional):
                _description_. Defaults to 0.1

            phase (float, optional):
                _description_. Defaults to 0.0

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

    def _stretch(
        self,
        frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Stretch a 2D image into a 1D vector.
        """

        # TODO: Handle transparency (4D tensors)?
        # if frame.dim() == 3:
        #     # Transpose the depth dimension and stretch
        #     return frame.permute(2, 1, 0).flatten()[:, None]

        # return frame.permute(1, 0).flatten()[:, None]
        return frame.T.flatten()

    def _fold(
        self,
        frame: pt.Tensor,
        h: int,
        w: int,
    ) -> pt.Tensor:
        """
        Fold a 1D vector into a 2D tensor.
        """

        # print(f"==[ frame shape: {frame.shape}")

        return frame.reshape(w, h).t()

    def _make_sparse_coordinates(
        self,
        w2: int,
        h2: int,
        rho_max: int,
        rho_fovea: int,
        inverse: bool = False,
    ):

        # Number of sectors
        # NOTE:
        # Ideally, this should be divisible by 4 because then the distribution
        # is nicely symmetric with respect to the x and y axes.
        # TODO:
        # Figure out if it's worth actually enforcing that the sectors be divisible by 4.
        S = self.sectors

        # Sector size
        q = S / (2 * pt.pi)

        # Growth factor for coupling S with R below.
        # This preserves the pixel aspect ratio.
        a = 1 + 1 / q

        # Number of radial rings (coupled with the number of sectors)
        # R = np.floor(1 / np.emath.logn(rho_max / rho_f, a)).astype(np.int32)
        log_a = math.log(a)
        R = math.floor(math.log(rho_max / rho_fovea) / log_a)

        # Cartesian coordinate mesh (x, y)
        rf_xs = pt.linspace(-h2, h2 + 1, self.h + 1)
        rf_ys = pt.linspace(-w2, w2 + 1, self.w + 1)
        Xs, Ys = pt.meshgrid(rf_xs, rf_ys, indexing="ij")

        # Logpolar coordinate mesh (r, φ)
        radii = pt.sqrt(Xs**2 + Ys**2)
        radial_ratios = Xs / radii
        angles = pt.arccos(radial_ratios)
        angles = pt.where(Ys > 0, 2 * pt.pi - angles, angles)

        # Eccentricity-dependent logpolar coordinates
        ksi = pt.log(radii / rho_fovea) / log_a
        eta = q * angles

        # Discrete versions of ksi and eta (mappable to pixel coordinates)
        u = pt.floor(ksi)
        v = pt.floor(eta)

        # Mesh of discrete logpolar coordinates of the RF centres
        # Rs: radii
        # Ts: angles
        rpoints = pt.unique(rho_fovea * (a**u))
        tpoints = pt.unique(v / q)
        (Rs, Ts) = pt.meshgrid(rpoints, tpoints, indexing="ij")

        # Convert Rs and Ts back to integer (x, y) coordinates
        rf_xs = pt.round(pt.cos(Ts) * Rs) + w2
        rf_ys = pt.round(pt.sin(Ts) * Rs) + h2

        # print(f"==[ {rf_xs.shape}")
        # print(f"==[ {rf_ys.shape}")

        # Eliminate duplicates
        coords = pt.unique(
            pt.stack(
                (rf_xs.flatten(), rf_ys.flatten()),
                dim=1,
            ),
            dim=0,
        ).int()

        # Trim coordinates that fall outside the image boundaries
        coords = self._trim(coords)

        centered_coords = coords - pt.tensor([w2, h2])
        rs = pt.sqrt(
            pt.pow(centered_coords[:, 0], 2) + pt.pow(centered_coords[:, 1], 2)
        )
        sorted = pt.argsort(rs, stable=True)
        coords = coords[sorted]

        # Maximal RF size
        if inverse:
            # TODO
            # This needs to be changed, right now
            # it's the same as the other branch
            max_rf_width = (rho_fovea / rho_max) * (a**R) * (1 - 1 / a)
        else:
            max_rf_width = (rho_fovea / rho_max) * (a**R) * (1 - 1 / a)

        return (coords, max_rf_width)

    def _make_eccentric_rfs(
        self,
        inverse: bool = False,
        dense: bool = False,
    ):
        """
        Implementation of eccentricity-dependent log-polar distribution
        of receptive fields as presented here, with slight corrections and improvements:

        Maiello, G., Chessa, M., Bex, P. J. & Solari, F.
        Near-optimal combination of disparity across a log-polar scaled visual field.
        PLoS Comput Biol 16, e1007699 (2020).

        Args:

            inverse (bool):
                Inverse log-polar distribution (larger RFs in the centre). Defaults to False.

            dense (bool):
                Create a dense coverage (one kernel per pixel). Defaults to False.
                    Note: Use this option with caution, especially for large layers.
                    It increases memory conumption substantially!
        """

        self.debug("Creating receptive fields...")

        # Coordinates of the central pixel
        w2 = self.w // 2
        h2 = self.h // 2

        # Construct the receptive fields
        # ==================================================

        # Maximal offset of the centre of the log-polar
        # rings from the centre of the FoV
        rho_max = math.sqrt(h2**2 + w2**2)

        # Size of the foveal region
        rho_fovea = (
            math.log(rho_max)
            if self.fovea_ratio is None
            else rho_max * self.fovea_ratio
        )

        # Prepare the coordinate mesh
        self._make_mesh()

        # Compute the coordinates and the RF size ratio
        if not dense:
            (coords, max_rf_width) = self._make_sparse_coordinates(
                w2, h2, rho_max, rho_fovea, inverse
            )
        else:
            # TODO: A dense version
            (coords, max_rf_width) = self._make_sparse_coordinates(
                w2, h2, rho_max, rho_fovea, inverse
            )

        # Make the receptive fields.
        # ==================================================
        rows = []
        cols = []
        values = []
        indices = []
        cell_coords = []
        neuron_count = 0

        extent = rho_max if self.extent is None else self.extent * rho_max

        for rx, ry in coords:
            cdist = pt.sqrt((rx - w2) ** 2 + (ry - h2) ** 2)
            # print(f"==[ {rx, ry}")

            if self.extent is not None and cdist > extent:
                continue

            if inverse:
                radius = max(
                    self.min_kernel_size,
                    self.kernel_scale * max_rf_width * (rho_max - cdist),
                )
                if radius < 3:
                    continue
            else:
                radius = max(
                    self.min_kernel_size, self.kernel_scale * max_rf_width * cdist
                )

            result = self.kernel_function(
                rx,
                ry,
                max(1, radius),
                **self.kernel_params,
            )

            if result is not None:
                (rs, cs, vs) = result

                rows.append(rs)
                cols.append(cs)
                values.append(vs)
                indices.append([len(indices) for _ in range(len(rs))])
                # rx = max(0, min(rx, self.h - 1))
                # ry = max(0, min(ry, self.w - 1))

                # These are the coordinates of the cell itself.
                # It should be in the centre of the receptive field.
                cell_coords.append([ry, rx])
                neuron_count += 1

                # print(f"==[ rs: {rs.shape}")
                # print(f"==[ {idx} ] indices: {len(indices[-1])}")

        # Prepare the sparse tensor coordinates and values
        concat_rows = np.concatenate(rows)
        concat_cols = np.concatenate(cols)
        sp_rows = np.concatenate(indices)
        sp_cols = concat_rows + self.h * concat_cols
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
                    self.dims.comp.span,
                ),
                dtype=conf.dtype,
            )
            .coalesce()
            .to_sparse_csr()
            .to(conf.device)
        )

        self.rows = rows
        self.cols = cols
        self.vals = values
        self.cell_coords = np.array(cell_coords)

        d = {(x, y): 0 for y in range(self.w) for x in range(self.h)}
        for coord in [
            tuple(_)
            for _ in np.vstack(
                [
                    concat_rows,
                    concat_cols,
                ]
            ).T
        ]:
            d[coord] += 1

        rf_factors = pt.zeros((self.h, self.w), dtype=conf.dtype)

        for (y, x), v in d.items():
            rf_factors[y, x] += v
        rf_factors[rf_factors == 0] = 1
        self.rf_factors = 1 / rf_factors

        self.rfs = rfs
        self.neuron_count = neuron_count

        self.debug(f"Receptive fields created for {self.neuron_count} neurons.")

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

        # wpo: width-wise pixel offset
        # hpo: height-wise pixel offset
        wpo = int(
            np.sign(width_offset) * math.floor(math.fabs(width_offset) * self.dim.W)
        )
        hpo = int(
            np.sign(height_offset) * math.floor(math.fabs(height_offset) * self.dim.H)
        )

        padding = tuple(
            self.dim.padding
            + np.array(
                [
                    hpo,
                    -hpo,
                    wpo,
                    -wpo,
                ]
            ).tolist()
        )

        return padding

    def _compute_ema(
        self,
        x: pt.Tensor,
        mean: pt.Tensor,
        alpha: pt.Tensor,
        *args,
        **kwargs,
    ):
        """
        Exponential runnning average (EMA).

        This is a shortcut version that does not create extra temporary variables.
        Note that we are not keeping track of the variance, however,
        the variance-tracking version is included below for future reference
        (`var` needs to be passed as a function argument).

        diff = x - mean
        inc = alpha * diff
        mean += inc
        sd = None

        if var is not None:
            var = (1.0 - alpha) * (var + diff * inc)
            sd = pt.sqrt(var + eps)

        return (mean, sd)
        """

        return alpha, mean + alpha * (x - mean)

    def _compute_iema(
        self,
        x: pt.Tensor,
        mean: pt.Tensor,
        alpha: float,
        dt: float,
        *args,
        **kwargs,
    ):
        """
        Irregular version of EMA.

        Here, we use the mean time period as the EMA range.

        References:

        - G. Zumbach and U. Müller (2001), ‘Operators on inhomogeneous time series’, Int. J. Theor. Appl. Finan., vol. 04, no. 01, pp. 147–177.
        """

        if dt <= 0.0:
            raise SystemExit("Non-monotonically increasing times detected")

        # Update the running stats for dt
        mean_dt = self._compute_ema(
            dt,
            self.mean_dt,
            alpha,
        )

        # Update alpha
        alpha = pt.exp(-dt / self.mean_dt)

        # Now return the regular EMA but with the new alpha.
        return self._compute_ema(x, mean, alpha)

    def convolve(
        self,
        frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Convolve the input frame with the current layer's receptive field.

        Args:
            frame (pt.Tensor):
                The input frame.

        Returns:
            pt.Tensor:
                Self-explanatory.
        """

        # This is the old behaviour using dense frames.
        # ==================================================
        # stretched = self._stretch(frame)
        # mean = pt.mv(self.rfs, stretched)
        # folded = self._fold(mean, self.h, self.w)
        # return folded

        # TODO: Investigate SciPy / CuPy
        # ==================================================
        return pt.mv(self.rfs, frame.T.flatten())
