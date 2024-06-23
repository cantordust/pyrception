from typing import *

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
import torch as pt

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
from pyrception import conf
from pyrception.logging import Logging
from pyrception.visual.util.types import RFArrangement
from pyrception.visual.util.types import KernelParams
from pyrception.visual.util.types import KernelFilter
from pyrception.visual.util.types import KernelShape


class ReceptiveFields(Logging):
    """
    A set of receptive fields for a specific substrate.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        substrate: pt.Tensor = None,
        sectors: int = 32,
        extent: int = 1.0,
        arrangement: RFArrangement = RFArrangement.LogPolar,
        fovea_ratio: int = 1.0,
        inverse: bool = False,
        dense: bool = False,
        compute_factors: bool = False,
        name: str = "Receptive fields",
        kernel_params: Dict[str, Any] = None,
    ):
        """

        Args:
            size (Tuple[int, ...]):
                The dimensions of the visual field (height, width, depth).
                NOTE: Colour vision is not implemented yet.
                It would be necessary to take into account the depth dimension.

            substrate (pt.Tensor):
                The coordinates of the input cells that constitute the 'substrate' to which
                the receptive fields are applied. These coordinates could be sparse.

            sectors (int, optional):
                Number of sectors ('wedges') for logpolar receptive fields. Defaults to 32.

            extent (int, optional):
                Extent of the receptive field coverage. Defaults to 1.0.
                A value of 1.0 means that the entire visual field is covered.

            arrangement (RFArrangement, optional):
                Defines how the RFs are arranged spatially to cover the visual field.
                Defaults to RFArrangement.LogPolar.

            fovea_ratio (int, optional):
                Ratio of the foveal region relative to the default size computed from the sectors.
                Defaults to 1.0.
                WIP This is not fully functional yet.

            inverse (bool, optional):
                Inverse distribution of RF sizes (i.e., RFs are larger in the centre and become
                smaller with eccentricity). Defaults to False.

            dense (bool, optional):
                Create dense coverage (one RF per pixel). Defaults to False.

                NOTE: Use this option with caution, especially for large layers.
                This may consume *a lot* of memory!

            compute_factors (bool, optional):
                Toggle indicating if receptive field factors should be computed. Defaults to False.
                The factors are used to average out the feedback of overlapping receptive fields.
                Mainly used for (inhibitory) feedback neurons, such as horizontal and amacrine cells.

            name (str, optional):
                Layer name. Defaults to "Receptive fields".
                Each derived class overrides the name with a default value.

            kernel_params (Dict[str, Any], optional):
                Parameters to pass to the kernel function. Defaults to None.
                This can be used to fine-tune the receptive fields.
                (cf. :class:`pyrception.visual.util.types.KernelParams` for details).
        """

        # Initialise the base
        super().__init__(name)

        if kernel_params is None:
            kernel_params = {}

        # TODO: add some assertions
        self.height = int(size[0])
        self.width = int(size[1])
        self.depth = int(size[2])
        self.sector_count = sectors
        self.extent = extent
        self.arrangement = arrangement
        self.fovea_ratio = fovea_ratio
        self.inverse = inverse
        self.dense = dense
        self.compute_factors = compute_factors
        self.kernel_params = KernelParams(**kernel_params)

        # Internal parameters used for constructing RFs.
        # ==================================================
        self.neuron_count = 0
        self.substrate = self._make_substrate() if substrate is None else substrate
        self.cell_coordinates = None
        self.rows = None
        self.cols = None
        self.vals = None
        self.rfs = None
        self.rf_factors = None
        self.rf_sizes = None
        self.rings = None
        self.sectors = None

    @property
    def _f_rf_arrangement(self) -> Callable:
        """
        Return the RF factory function based on the RF arrangement.

        Raises:
            TypeError:
                Raised if the requested RF arrangement is invalid.

        Returns:
            Callable:
                The RF factory function for the requested RF arrangement.
        """

        functions = {
            RFArrangement.LogPolar: self._make_logpolar_rfs,
            RFArrangement.Cartesian: self._make_cartesian_rfs,
        }

        function = functions.get(self.arrangement, None)

        if function is None:
            raise TypeError(f"Invalid RF arrangement '{self.arrangement}'")

        return function

    @property
    def _f_kernel_shape(self) -> Callable:
        """
        Return the kernel factory function based on the kernel shape.

        Raises:
            TypeError:
                Raised if the requested kernel shape is invalid.

        Returns:
            Callable:
                The kernel factory function for the requested kernel shape.
        """

        functions = {
            KernelShape.Elliptic: self._make_elliptic_kernel,
            KernelShape.Rectangular: self._make_rectangular_kernel,
        }

        function = functions.get(self.kernel_params.shape, None)

        if function is None:
            raise TypeError(f"Invalid kernel shape '{self.kernel_params.shape}'")

        return function

    @property
    def _f_kernel_filter(self) -> Callable:
        """
        Return the kernel factory function based on the kernel filter.

        Raises:
            TypeError:
                Raised if the requested kernel filter is invalid.

        Returns:
            Callable:
                The kernel factory function for the requested kernel filter.
        """

        functions = {
            KernelFilter.Flat: self._make_flat_kernel,
            KernelFilter.Gaussian: self._make_gaussian_kernel,
            KernelFilter.Gabor: self._make_gabor_kernel,
        }

        function = functions.get(self.kernel_params.filter, None)

        if function is None:
            raise TypeError(f"Invalid kernel filter '{self.kernel_params.filter}'")

        return function

    def _make_substrate(
        self,
        step: int = 1,
    ):
        """
        Create a Cartesian coordinate mesh with all possible combinations
        of widths and heights (= columns and rows). These are the coordinates
        of all the pixels in the raw input.

        Args:
            step (int, optional):
                Grid step (defines the coarseness of the mesh). Defaults to 1.
        """

        # A mesh of all possible coordinate pairs
        rows = pt.linspace(0, self.height - 1, self.height // step, dtype=pt.int32)
        cols = pt.linspace(0, self.width - 1, self.width // step, dtype=pt.int32)
        return pt.cartesian_prod(rows, cols)

    def _make_elliptic_kernel(
        self,
        substrate: pt.Tensor,
        centre: pt.Tensor,
        spread: pt.Tensor,
        angle: float = 0.0,
    ) -> Tuple[pt.Tensor, ...]:
        """
        Create an elliptic kernel centred at a certain pixel and
        having the specified spread (semi-major axes).
        The kernel can be optionally rotated at an angle.

        Args:
            substrate (pt.Tensor):
                The coordinate substrate.

            centre (np.ndarray):
                The centre of the ellipse

            spread (np.ndarray):
                Spread of the ellipse.
                This is an array of two numbers representing the two
                semi-major axes along the x and y dimensions.

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

        Returns:
            Tuple[pt.Tensor, ...]:
                A tuple containing:
                    1. Coordinate columns.
                    2. Coordinate rows.
                    3. Coordinate indices.
                    4. Pixel coordinates along the width dimension.
                    5. Pixel coordinates along the height dimension.

                NOTE: The last two are used for computing the kernel weights
                (for Gaussian and flat kernels).
        """

        # Columns and rows offset to the given centre coordinates.
        rows = substrate[:, 0] - centre[0]
        cols = substrate[:, 1] - centre[1]

        # Apply the rotation matrix.
        sin = np.sin(angle)
        cos = np.cos(angle)
        xs = -cols * sin + rows * cos
        ys = cols * cos + rows * sin

        # Extract the indices that fall within the ellipse.
        k_idx = pt.argwhere((xs / spread[0]) ** 2 + (ys / spread[1]) ** 2 < 1)[:, 0]

        # Extract the coordinates of the kernel from the substrate as row / column pairs.
        coords = substrate[k_idx]
        (k_rows, k_cols) = (coords[:, 0], coords[:, 1])

        return (k_rows, k_cols, k_idx, xs[k_idx], ys[k_idx])

    def _make_rectangular_kernel(
        self,
        substrate: pt.Tensor,
        centre: np.ndarray,
        spread: np.ndarray,
        scale: float = 1.0,
        angle: float = 0.0,
    ) -> Tuple[pt.Tensor, ...]:
        """
        Create a rectangular kernel centred at a certain pixel and
        having the specified spread (side lengths).
        The kernel can be optionally rotated at an angle.

        WIP: This is a stub - to be implemented.

        Args:
            substrate (pt.Tensor):
                The coordinate substrate.

            centre (np.ndarray):
                The centre of the rectangle (the crossing point of its diagonals).

            spread (np.ndarray):
                Spread of the rectangle.
                This is an array of two numbers representing the lenghts of
                the two sides along the x and y dimensions.

            scale (float, optional):
                Scale of the kernel relative to the default. Defaults to 1.0.

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

        Returns:
            Tuple[pt.Tensor, ...]:
                A tuple containing:
                    1. Coordinate columns.
                    2. Coordinate rows.
                    3. Coordinate indices.
                    4. Pixel coordinates along the width dimension.
                    5. Pixel coordinates along the height dimension.

                NOTE: The last two are used for computing the kernel weights
                (for Gaussian and flat kernels).
        """
        pass

    def _make_flat_kernel(
        self,
        substrate: pt.Tensor,
        centre: Tuple[int, int],
        spread: Tuple[int, int],
        angle: float = 0.0,
        weights: Union[pt.Tensor, float] = None,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        2D flat kernel with a given centre, spread and rotation angle.

        Args:
            substrate (pt.Tensor):
                Coordinate substrate as a tensor with shape (N, 2).
                Could be sparse.

            centre (Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            weights (Union[pt.Tensor, float], optional):
                Weights of the kernel. Defaults to None.
                If provided as a `float`, all kernels will have the same weight.

            shape (str, optional):
                kernel shape. Defaults to "elliptic".

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        # Get the rows and columns for the kernel
        (rows, cols, idx, _, _) = self._f_kernel_shape(
            substrate,
            centre,
            spread,
            angle,
        )

        # Bail out if the kernel size is 0
        if cols.numel() == 0:
            return

        if weights is None:
            weights = 1 / cols.numel()

        vals = pt.full_like(cols, weights, dtype=pt.float32)

        # Return the kernel indices and values
        return (rows, cols, vals, idx)

    def _make_gaussian_kernel(
        self,
        substrate: pt.Tensor,
        centre: pt.Tensor,
        spread: pt.Tensor,
        angle: float = 0.0,
        sd: Union[Tuple[float, ...], float] = (0.37, 0.37),
        normalise: bool = True,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        2D Gaussian kernel with a given mean, SD and rotation angle.

        Args:
            substrate (pt.Tensor):
                Coordinate substrate as a tensor with shape (N, 2).
                Could be sparse.

            centre (Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            sd (Union[Tuple[float, ...], float], optional):
                Standard deviation of the kernel in x and y directions
                as a percentage of the spread. Defaults to (0.37, 0.37).
                This is approximately one standard deviation.

            shape (str, optional):
                Kernel shape. Defaults to "elliptic".

            normalise (bool, optional):
                Toggle for Gaussian normalisation. Defaults to True.

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        # Get the rows and columns for the kernel
        (rows, cols, idx, xs, ys) = self._f_kernel_shape(
            substrate,
            centre,
            spread,
            angle,
        )

        # Bail out if the kernel size is 0
        if cols.numel() == 0:
            return

        # Compute the standard deviation relative to the spread
        if isinstance(sd, float):
            sd = [sd, sd]
        sd = pt.tensor(sd, dtype=pt.float32)
        sd *= spread

        # Create a Gaussian distribution
        vals = pt.exp(-0.5 * ((xs / sd[0]) ** 2 + (ys / sd[1]) ** 2))

        # Normalise if necessary
        if normalise:
            vals /= 2 * pt.pi * (sd[0] * sd[1])

        # Return the kernel parameters.
        return (rows, cols, vals, idx)

    def _make_gabor_kernel(
        self,
        substrate: pt.Tensor,
        centre: pt.Tensor,
        spread: pt.Tensor,
        angle: float = 0.0,  # Orientation [deg]
        sd: Union[Tuple[float, ...], float] = (0.37, 0.37),
        frequency: float = 0.1,  # Sine component frequency
        aspect: float = 0.1,  # Aspect ratio
        phase: float = 0.0,  # Phase of the sine component
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Gabor filters.

        WIP: This is just a stub.
        TODO: Check if we can incorporate this into the Gaussian kernel method
        since we only have to add a couple of extra parameters and superimpose
        the sine funciton.

        Args:
            substrate (pt.Tensor):
                Coordinate substrate as a tensor with shape (N, 2).
                Could be sparse.

            centre (Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            sd (Union[Tuple[float, ...], float], optional):
                Standard deviation of the kernel in x and y directions
                as a percentage of the spread. Defaults to (0.37, 0.37).
                This is approximately one standard deviation.

            frequency (float, optional):
                Sine frequency. Defaults to 0.1

            aspect (float, optional):
                Aspect. Defaults to 0.1

            phase (float, optional):
                Sine phase. Defaults to 0.0

        Returns:
            Tuple[pt.Tensor, pt.Tensor]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """
        pass

    def _trim_to_vf(
        self,
        coordinates: pt.Tensor,
    ) -> Tuple[pt.Tensor, ...]:
        """
        Trim the coordinates of the receptive fields to the
        dimensions of the visual field.

        Args:
            coordinates (pt.Tensor):
                Pixel coordinates of the receptive field.

        Returns:
            Tuple[pt.Tensor, ...]:
                A tuple containing:
                    1. The trimmed coordinates.
                    2. The mask that trims the coordinates to the visual field
                    3. An index array for the subset of unique coordinates
                        (since some coordinates might be repeated).
        """

        # Mask to trim the indices to the visual field.
        vf_mask = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 0] < self.height)
            & (coordinates[:, 1] < self.width)
        )

        # REVIEW
        # This is a hack since the PyTorch version of
        # unique() does not return the unique indices
        # like NumPy does.
        #
        # Rework once this is implemented in PyTorch.
        # Cf. https://github.com/pytorch/pytorch/issues/36748
        # ==================================================
        coordinates = coordinates[vf_mask]

        unique_indices = pt.tensor(
            sorted(
                np.unique(
                    coordinates.numpy(),
                    return_index=True,
                    axis=0,
                )[1]
            ),
            dtype=pt.int32,
        )

        return (coordinates[unique_indices], vf_mask, unique_indices)

    def _extract_segment_indices(
        self,
        container: Iterable,
    ) -> List[pt.Tensor]:
        """
        Extract the indices of a specific segment.
        For now, this is limited to rings and sectors,
        but it should work with any type of segment
        (e.g., row, column, kernel...) as long as the
        coordinate indices are sorted appropriately.

        Args:
            container (Iterable):
                An iterable containing tuples of some metric
                (e.g., radius, angle, etc.) and its index.
                The container should be sorted in such a way
                that each segment is represented by a continuous
                portion of the container.

        Returns:
            List[pt.Tensor]:
                The indices of the coordinates of the segment.
        """

        segments = []
        segment = []
        cur_item = -1.0
        for idx, (item, index) in enumerate(container):
            if item < cur_item:
                segments.append(pt.tensor(segment, dtype=pt.int32))
                segment = [index]
                cur_item = -1.0
            else:
                segment.append(index)
                cur_item = item

            if idx == len(container) - 1 and len(segment) > 0:
                segments.append(pt.tensor(segment, dtype=pt.int32))

        return segments

    def _segment(
        self,
        logpolar: pt.Tensor,
        mask: pt.Tensor,
        indices: pt.Tensor,
    ):

        # Mask and index the logpolar coordinates
        # and convert them into a NumPy array.
        # This would be convenient later for computing
        # the sectors.
        logpolar = logpolar[mask][indices].numpy()
        indices = pt.arange(logpolar.shape[0])

        # Extract the coordinates of each ring.
        # The logpolar coordinates are sorted by angle,
        # so we can just iterate over the array.
        # ==================================================
        ring_indices = list(zip(logpolar[:, 1], indices))
        self.rings = self._extract_segment_indices(ring_indices)

        # Extract the coordinates of each sector.
        # Sort the polar coordinates by radius.
        # Before sorting, we zip the array with an ordered
        # array so that we can access the correct index
        # once the array is sorted.
        # ==================================================
        sector_indices = [
            (item[0][0], item[1])
            for item in sorted(zip(logpolar, indices), key=lambda x: x[0][1])
        ]
        self.sectors = self._extract_segment_indices(sector_indices)

    def _compute_sparse_coordinates(
        self,
        rho_max: int,
        rho_fovea: int,
    ) -> Tuple[pt.Tensor, int]:
        """
        Create the sparse coordinates for the receptive fields.

        REVIEW
        This method assumes that we are working with logpolar coordinates.
        It needs to be split into multiple methods that can produce
        different types of receptive field distributions.

        References:
            Maiello, G., Chessa, M., Bex, P. J. & Solari, F.
            Near-optimal combination of disparity across a log-polar scaled visual field.
            PLoS Comput Biol 16, e1007699 (2020).

        Args:

            rho_max (int):
                Maximal size of any receptive field.

            rho_fovea (int):
                Maximal size of a receptive field in the fovea.
                This should be 1 by default, but it can be overridden,
                for instance, for horizontal cells, which compute the
                local (spatial) mean illumination for the adjacent receptors.
                These should ideally encompass multiple receptors, otherwise
                the spatial mean illumination would make no sense.

        Returns:
            Tuple[pt.Tensor, int]:
                A tuple containing:
                    1. The sparse coordinates of the receptive fields.
                    2. The maximal size of the receptive fields.
        """
        # Coordinates of the central pixel
        h2 = self.height // 2
        w2 = self.width // 2

        # Number of sectors.
        #
        # NOTE
        # Ideally, this should be divisible by 4 because
        # then the distribution is nicely symmetric with
        # respect to the x and y axes.
        #
        # TODO
        # Figure out if we should actually *enforce* that
        # the sectors be divisible by 4.
        # ==================================================
        S = self.sector_count

        # Sector size.
        q = S / (2 * pt.pi)

        # Growth factor for coupling S with R below.
        # This preserves the pixel aspect ratio.
        a = 1 + 1 / q

        # Number of radial rings (coupled with the number of sectors)
        log_a = math.log(a)
        R = math.floor(math.log(rho_max / rho_fovea) / log_a)

        # Cartesian coordinate mesh (x, y)
        rf_rows = pt.arange(self.height) - h2
        rf_cols = pt.arange(self.width) - w2
        cart_prod = pt.cartesian_prod(rf_rows, rf_cols)
        (Rs, Cs) = cart_prod[:, 0], cart_prod[:, 1]

        # Logpolar coordinate mesh (r, φ)
        radii = pt.sqrt(Rs**2 + Cs**2)
        radial_ratios = Rs / radii
        angles = pt.arccos(radial_ratios)
        angles = pt.where(Cs > 0, 2 * pt.pi - angles, angles)

        # Eccentricity-dependent logpolar coordinates of the cells
        # ξ: Radius
        # η: Angle
        ksi = pt.log(radii / rho_fovea) / log_a
        eta = q * angles

        # Floored versions of ξ and η (mappable to pixel coordinates)
        u = pt.floor(ksi)
        v = pt.floor(eta)

        # Mesh of discrete logpolar coordinates of the RF centres
        # Rs: Radii
        # Ts: Angles
        rf_radii = pt.unique(rho_fovea * (a**u))
        rf_angles = pt.unique(v / q)

        polar_cart_prod = pt.cartesian_prod(rf_radii, rf_angles)
        (Rs, Ts) = polar_cart_prod[:, 0], polar_cart_prod[:, 1]

        # Convert Rs and Ts back to integer (x, y) coordinates
        rf_rows = (pt.round(pt.sin(Ts) * Rs) + h2).type(pt.int32)
        rf_cols = (pt.round(pt.cos(Ts) * Rs) + w2).type(pt.int32)

        # Create the (x, y) coordinates
        coords = pt.stack((rf_rows, rf_cols), dim=1)

        # Trim coordinates that fall outside the image boundaries
        (coords, mask, indices) = self._trim_to_vf(coords)

        # Segment the coordinates into rings and sectors
        self._segment(polar_cart_prod, mask, indices)

        # Compute the maximal RF size
        # ==================================================
        if self.inverse:
            # TODO
            # This needs to be changed.
            # Right now, it's the same as the else branch,
            max_rf_size = (rho_fovea / rho_max) * (a**R) * (1 - 1 / a)
        else:
            max_rf_size = (rho_fovea / rho_max) * (a**R) * (1 - 1 / a)

        return (coords, max_rf_size)

    def _make_logpolar_rfs(self):
        """
        Implementation of eccentricity-dependent log-polar distribution
        of receptive fields as presented in the following paper,
        with slight corrections and improvements:

        Maiello, G., Chessa, M., Bex, P. J. & Solari, F.
        Near-optimal combination of disparity across a log-polar scaled visual field.
        PLoS Comput Biol 16, e1007699 (2020).
        """

        self.debug("Creating receptive fields...")

        # Coordinates of the central pixel
        h2 = self.height // 2
        w2 = self.width // 2

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

        # Compute the coordinates and the RF size ratio
        if self.dense:
            # TODO: Design a proper dense version
            (coords, max_rf_size) = self._compute_sparse_coordinates(rho_max, rho_fovea)
        else:
            (coords, max_rf_size) = self._compute_sparse_coordinates(rho_max, rho_fovea)

        # Make the receptive fields.
        # ==================================================
        rows = []
        cols = []
        rf_sizes = []
        cell_coordinates = []
        neuron_count = 0

        sparse_rows = []
        sparse_cols = []
        sparse_vals = []

        # Extent of the receptive fields as a portion of the maximum.
        self.extent *= rho_max

        # Distances from the central pixel.
        distances = pt.sqrt((coords[:, 0] - h2) ** 2 + (coords[:, 1] - w2) ** 2)

        for rf_centre, distance in tqdm(
            zip(coords, distances),
            total=len(distances),
            desc="Neurons created: ",
        ):

            # Do not proceed if the distance is greater
            # than the extent.
            # ==================================================
            if self.extent is not None and distance > self.extent:
                continue

            # Check if the receptive fields should shrink
            # (instead of growing) with eccentricity.
            # ==================================================
            if self.inverse:
                rf_size = self.kernel_params.scale * max_rf_size * (rho_max - distance)
            else:
                rf_size = self.kernel_params.scale * max_rf_size * distance

            rf_size = pt.tensor([rf_size, rf_size])

            # RF spread.
            # The spread is assumed to represent the full size
            # of the RF (e.g., the semi-major axes in the
            # case of an elliptic RF).
            # We have to scale and then halve it to make sure
            # that the RF factories work as expected.
            # ==================================================
            rf_spread = pt.max(
                pt.vstack((rf_size, self.kernel_params.min_size)), dim=0
            )[0]
            rf_spread *= self.kernel_params.scale * self.kernel_params.aspect

            # Create the RF
            # ==================================================
            result = self._f_kernel_filter(
                self.substrate,
                rf_centre,  # Centre
                rf_spread,
                **self.kernel_params.params,  # Additional RF parameters
            )

            # Process the results
            # ==================================================
            if result is not None:
                # Unpack the result
                (rf_rows, rf_cols, rf_vals, rf_idx) = result

                # Update the rows, columns and values
                rows.append(rf_rows)
                cols.append(rf_cols)
                # indices.append(pt.full_like(k_rows, len(indices)))
                sparse_rows.append(pt.full_like(rf_idx, len(sparse_rows)))
                sparse_cols.append(rf_idx)
                sparse_vals.append(rf_vals)

                # Store the coordinates of the cell itself.
                # It should be in the centre of the receptive field.
                cell_coordinates.append(rf_centre)

                # Store the size of the RF of the cell
                rf_sizes.append(rf_size)
                neuron_count += 1

        # Prepare the indices and the values of
        # the sparse tensor
        # ==================================================
        sparse_rows = pt.concatenate(sparse_rows)
        sparse_cols = pt.concatenate(sparse_cols)
        sparse_vals = pt.concatenate(sparse_vals)

        sparse_indices = pt.vstack(
            [
                sparse_rows,
                sparse_cols,
            ]
        )

        # Create the actual receptive fields
        # ==================================================
        self.rfs = (
            pt.sparse_coo_tensor(
                sparse_indices,
                sparse_vals,
                # Size of the dense tensor, necessary for correct multiplication
                size=(
                    neuron_count,  # rows
                    len(self.substrate),  # columns
                ),
                dtype=conf.dtype,
            )
            .coalesce()
            .to_sparse_csr()
            .to(conf.device)
        )

        # Store various useful tensors and scalars.
        #
        # NOTE
        # These might be redundant, but for now they
        # can be used for visualisation purposes.
        # ==================================================
        self.rows = rows
        self.cols = cols
        self.vals = sparse_vals
        self.cell_coordinates = pt.vstack(cell_coordinates)
        self.rf_sizes = pt.vstack(rf_sizes)
        self.neuron_count = neuron_count
        self.rf_coords = pt.vstack(
            (
                pt.concatenate(rows),
                pt.concatenate(cols),
            )
        )

        # Compute the RF weight factors.
        # This is necessary for handling overlapping RFs,
        # for instance, in the case of horizontal cells.
        # ==================================================
        if self.compute_factors:
            (unique_rf_coords, occurrences) = self.rf_coords.unique(
                return_counts=True, dim=1
            )
            rf_factors = pt.ones(
                (self.height, self.width),
                dtype=conf.dtype,
                device=conf.device,
            )

            rf_factors[
                unique_rf_coords[0],
                unique_rf_coords[1],
            ] = occurrences.type(conf.dtype)
            self.rf_factors = (1 / rf_factors).flatten()

        self.debug(f"Receptive fields created for {self.neuron_count} neurons.")

    def _make_cartesian_rfs(self):
        """
        TODO: Implement this. :)
        """
        pass

    def make_rfs(self, *args, **kwargs):
        return self._f_rf_arrangement(*args, **kwargs)
