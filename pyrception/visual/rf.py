import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
from scipy.sparse import csc_array

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import pyrception.util.functions as pcf
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
        shape: tp.Tuple[int, ...],
        substrate: np.ndarray = None,
        sectors: int = 64,
        extent: int = 1.0,
        arrangement: RFArrangement = RFArrangement.LogPolar,
        inverse: bool = False,
        dense: bool = False,
        create_feedback: bool = False,
        name: str = "Receptive fields",
        kernel_params: tp.Dict[str, tp.Any] = None,
    ):
        """

        Args:
            shape (tp.Tuple[int, ...]):
                The dimensions of the visual field (height, width, depth).
                NOTE: Colour vision is not implemented yet.
                It would be necessary to take into account the depth dimension.

            substrate (np.ndarray):
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

            inverse (bool, optional):
                Inverse distribution of RF sizes (i.e., RFs are larger in the centre and become
                smaller with eccentricity). Defaults to False.

            dense (bool, optional):
                Create dense coverage (one RF per pixel). Defaults to False.

                NOTE: Use this option with caution, especially for large layers.
                This may consume *a lot* of memory!

            create_feedback (bool, optional):
                Toggle indicating if receptive field factors should be computed. Defaults to False.
                The factors are used to average out the feedback of overlapping receptive fields.
                Mainly used for (inhibitory) feedback neurons, such as horizontal and amacrine cells.

            name (str, optional):
                Layer name. Defaults to "Receptive fields".
                Each derived class overrides the name with a default value.

            kernel_params (tp.Dict[str, tp.Any], optional):
                Parameters to pass to the kernel function. Defaults to None.
                This can be used to fine-tune the receptive fields.
                (cf. :class:`pyrception.visual.util.types.KernelParams` for details).
        """

        # Initialise the base
        super().__init__(name)

        if kernel_params is None:
            kernel_params = {}

        # TODO: add some assertions
        self.height = int(shape[0])
        self.width = int(shape[1])
        self.depth = int(shape[2])
        self.sector_count = sectors
        self.extent = extent
        self.arrangement = arrangement
        self.inverse = inverse
        self.dense = dense
        self.create_feedback = create_feedback
        self.kernel_params = KernelParams(**kernel_params)

        # Internal parameters used for constructing RFs.
        # ==================================================
        self.neuron_count = 0
        self.substrate = self._make_substrate() if substrate is None else substrate
        self.forward_synapses = None
        self.feedback_synapses = None
        self.rf_rows = None
        self.rf_cols = None
        self.rf_vals = None
        self.rf_sizes = None
        self.cell_rows = None
        self.cell_cols = None
        self.cell_rings = None
        self.cell_sectors = None
        self.cell_coordinates = None

    @property
    def _rf_arrangement_dispather(self) -> tp.Callable:
        """
        Return the RF factory function based on the RF arrangement.

        Raises:
            TypeError:
                Raised if the requested RF arrangement is invalid.

        Returns:
            tp.Callable:
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
    def _kernel_shape_dispather(self) -> tp.Callable:
        """
        Return the kernel factory function based on the kernel shape.

        Raises:
            TypeError:
                Raised if the requested kernel shape is invalid.

        Returns:
            tp.Callable:
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
    def _kernel_filter_dispatcher(self) -> tp.Callable:
        """
        Return the kernel factory function based on the kernel filter.

        Raises:
            TypeError:
                Raised if the requested kernel filter is invalid.

        Returns:
            tp.Callable:
                The kernel factory function for the requested kernel filter.
        """

        functions = {
            KernelFilter.Uniform: self._make_uniform_kernel,
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
        rows = np.linspace(0, self.height - 1, self.height // step, dtype=np.int32)
        cols = np.linspace(0, self.width - 1, self.width // step, dtype=np.int32)
        return pcf.cartesian_prod(rows, cols)

    def _make_elliptic_kernel(
        self,
        centre: np.ndarray,
        spread: np.ndarray,
        angle: float = 0.0,
        substrate: np.ndarray = None,
    ) -> tp.Tuple[np.ndarray, ...]:
        """
        Create an elliptic kernel centred at a certain pixel and
        having the specified spread (semi-major axes).
        The kernel can be optionally rotated at an angle.

        Args:

            centre (np.ndarray):
                The centre of the ellipse

            spread (np.ndarray):
                Spread of the ellipse.
                This is an array of two numbers representing the two
                semi-major axes along the x and y dimensions.

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            substrate (np.ndarray):
                Coordinate substrate as an array with a shape of (N, 2).
                Can be sparse.

        Returns:
            tp.Tuple[np.ndarray, ...]:
                A tuple containing:
                    1. Coordinate columns.
                    2. Coordinate rows.
                    3. Coordinate indices.
                    4. Pixel coordinates along the width dimension.
                    5. Pixel coordinates along the height dimension.

                NOTE: The last two are used for computing the kernel weights
                (for Gaussian and uniform kernels).
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
        k_idx = np.argwhere((xs / spread[0]) ** 2 + (ys / spread[1]) ** 2 < 1)[:, 0]

        # Extract the coordinates of the kernel from the substrate as row / column pairs.
        coords = substrate[k_idx]
        (k_rows, k_cols) = (coords[:, 0], coords[:, 1])

        return (k_rows, k_cols, k_idx, xs[k_idx], ys[k_idx])

    def _make_rectangular_kernel(
        self,
        centre: np.ndarray,
        spread: np.ndarray,
        angle: float = 0.0,
        substrate: np.ndarray = None,
    ) -> tp.Tuple[np.ndarray, ...]:
        """
        Create a rectangular kernel centred at a certain pixel and
        having the specified spread (side lengths).
        The kernel can be optionally rotated at an angle.

        WIP: This is a stub - to be implemented.

        Args:

            centre (np.ndarray):
                The centre of the rectangle (the crossing point of its diagonals).

            spread (np.ndarray):
                Spread of the rectangle.
                This is an array of two numbers representing the lenghts of
                the two sides along the x and y dimensions.

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            substrate (np.ndarray):
                Coordinate substrate as an array with a shape of (N, 2).
                Can be sparse.

        Returns:
            tp.Tuple[np.ndarray, ...]:
                A tuple containing:
                    1. Coordinate columns.
                    2. Coordinate rows.
                    3. Coordinate indices.
                    4. Pixel coordinates along the width dimension.
                    5. Pixel coordinates along the height dimension.

                NOTE: The last two are used for computing the kernel weights
                (for Gaussian and uniform kernels).
        """

        if substrate is None:
            substrate = self.substrate

        # Columns and rows offset to the given centre coordinates.
        rows = substrate[:, 0] - centre[0]
        cols = substrate[:, 1] - centre[1]

        # Apply the rotation matrix.
        sin = np.sin(angle)
        cos = np.cos(angle)
        xs = -cols * sin + rows * cos
        ys = cols * cos + rows * sin

        # Extract the indices that fall within the rectangle.
        k_idx = np.argwhere(
            (abs(xs) < spread[0] / 2)
            & (abs(ys) < spread[1] / 2)
        )[:, 0]

        # Extract the coordinates of the kernel from the substrate as row / column pairs.
        coords = substrate[k_idx]
        (k_rows, k_cols) = (coords[:, 0], coords[:, 1])

        return (k_rows, k_cols, k_idx, xs[k_idx], ys[k_idx])

    def _make_uniform_kernel(
        self,
        centre: tp.Tuple[int, int],
        spread: tp.Tuple[int, int],
        angle: float = 0.0,
        weights: tp.Union[np.ndarray, float] = None,
        substrate: np.ndarray = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        2D uniform kernel with a given centre, spread and rotation angle.

        Args:

            centre (tp.Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (tp.Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            weights (tp.Union[np.ndarray, float], optional):
                Weights of the kernel. Defaults to None.
                If provided as a `float`, all kernels will have the same weight.

            substrate (np.ndarray):
                Coordinate substrate as an array with a shape of (N, 2).
                Can be sparse.

        Returns:
            tp.Tuple[np.ndarray, np.ndarray]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        if substrate is None:
            substrate = self.substrate

        # Get the rows and columns for the kernel
        (rows, cols, idx, _, _) = self._kernel_shape_dispather(
            centre,
            spread,
            angle,
            substrate,
        )

        # Bail out if the kernel size is 0
        if cols.size == 0:
            return

        if weights is None:
            weights = 1 / cols.size

        vals = np.full_like(cols, weights, dtype=np.float32)

        # Return the kernel indices and values
        return (rows, cols, vals, idx)

    def _make_gaussian_kernel(
        self,
        centre: np.ndarray,
        spread: np.ndarray,
        angle: float = 0.0,
        sd: tp.Union[tp.Tuple[float, ...], float] = (0.37, 0.37),
        normalise: bool = True,
        substrate: np.ndarray = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        2D Gaussian kernel with a given mean, SD and rotation angle.

        Args:

            centre (tp.Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (tp.Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            sd (tp.Union[tp.Tuple[float, ...], float], optional):
                Standard deviation of the kernel in x and y directions
                as a percentage of the spread. Defaults to (0.37, 0.37).
                This is approximately one standard deviation.

            normalise (bool, optional):
                Toggle for Gaussian normalisation. Defaults to True.

            substrate (np.ndarray):
                Coordinate substrate as an array with a shape of (N, 2).
                Can be sparse.

        Returns:
            tp.Tuple[np.ndarray, np.ndarray]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        if substrate is None:
            substrate = self.substrate

        # Get the rows and columns for the kernel
        (rows, cols, idx, xs, ys) = self._kernel_shape_dispather(
            centre,
            spread,
            angle,
            substrate,
        )

        # Bail out if the kernel size is 0
        if cols.size == 0:
            return

        # Compute the standard deviation relative to the spread
        if isinstance(sd, float):
            sd = [sd, sd]
        sd = np.array(sd, dtype=np.float32)
        sd *= spread

        # Create a Gaussian distribution
        vals = np.exp(-0.5 * ((xs / sd[0]) ** 2 + (ys / sd[1]) ** 2))

        # Normalise if necessary
        if normalise:
            vals /= 2 * np.pi * (sd[0] * sd[1])

        # Return the kernel parameters.
        return (rows, cols, vals, idx)

    def _make_gabor_kernel(
        self,
        centre: np.ndarray,
        spread: np.ndarray,
        angle: float = 0.0,  # Orientation [deg]
        sd: tp.Union[tp.Tuple[float, ...], float] = (0.37, 0.37),
        frequency: float = 0.1,  # Sine component frequency
        aspect: float = 0.1,  # Aspect ratio
        phase: float = 0.0,  # Phase of the sine component
        substrate: np.ndarray = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Gabor filters.

        WIP: This is just a stub.
        TODO: Check if we can incorporate this into the Gaussian kernel method
        since we only have to add a couple of extra parameters and superimpose
        the sine funciton.

        Args:

            centre (tp.Tuple[int, float]):
                Coordinates of the centre of the kernel.

            spread (tp.Tuple[int, float]):
                Spread of the kernel (e.g., semi-major axes in the case of elliptic kernels).

            angle (float, optional):
                Rotation angle. Defaults to 0.0.

            sd (tp.Union[tp.Tuple[float, ...], float], optional):
                Standard deviation of the kernel in x and y directions
                as a percentage of the spread. Defaults to (0.37, 0.37).
                This is approximately one standard deviation.

            frequency (float, optional):
                Sine frequency. Defaults to 0.1

            aspect (float, optional):
                Aspect. Defaults to 0.1

            phase (float, optional):
                Sine phase. Defaults to 0.0

            substrate (np.ndarray):
                Coordinate substrate as an array with a shape of (N, 2).
                Can be sparse.

        Returns:
            tp.Tuple[np.ndarray, np.ndarray]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        if substrate is None:
            substrate = self.substrate

    def _crop_to_vf(
        self,
        coordinates: np.ndarray,
    ) -> tp.Tuple[np.ndarray, ...]:
        """
        Trim the coordinates of the receptive fields to the
        dimensions of the visual field.

        Args:
            coordinates (np.ndarray):
                Pixel coordinates of the receptive field.

        Returns:
            tp.Tuple[np.ndarray, ...]:
                A tuple containing:
                    1. The trimmed coordinates.
                    2. The mask that trims the coordinates to the visual field.
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

        # Return unique indices.
        # ==================================================
        coordinates = coordinates[vf_mask]

        unique_indices = np.array(
            sorted(
                np.unique(
                    coordinates,
                    return_index=True,
                    axis=0,
                )[1]
            ),
            dtype=np.int32,
        )

        return (coordinates[unique_indices], vf_mask, unique_indices)

    def _extract_segment_indices(
        self,
        container: tp.Iterable,
    ) -> tp.List[np.ndarray]:
        """
        Extract the indices of a specific segment.
        For now, this is limited to rings and sectors,
        but it should work with any type of segment
        (e.g., row, column, kernel...) as long as the
        coordinate indices are sorted appropriately.

        Args:
            container (tp.Iterable):
                An iterable containing tuples of some metric
                (e.g., radius, angle, etc.) and its index.
                The container should be sorted in such a way
                that each segment is represented by a continuous
                portion of the container.

        Returns:
            tp.List[np.ndarray]:
                The indices of the coordinates of the segment.
        """

        segments = []
        segment = []
        cur_item = -1.0
        for idx, (item, index) in enumerate(container):
            if item < cur_item:
                segments.append(np.array(segment, dtype=np.int32))
                segment = [index]
                cur_item = -1.0
            else:
                segment.append(index)
                cur_item = item

            if idx == len(container) - 1 and len(segment) > 0:
                segments.append(np.array(segment, dtype=np.int32))

        return segments

    def _segment(
        self,
        logpolar: np.ndarray,
        mask: np.ndarray,
        indices: np.ndarray,
    ):

        # Mask and index the logpolar coordinates
        # and convert them into a NumPy array.
        # This would be convenient later for computing
        # the sectors.
        logpolar = logpolar[mask][indices]
        indices = np.arange(logpolar.shape[0])

        # Extract the coordinates of each ring.
        # The logpolar coordinates are sorted by angle,
        # so we can just iterate over the array.
        # ==================================================
        ring_indices = list(zip(logpolar[:, 1], indices))
        self.cell_rings = self._extract_segment_indices(ring_indices)

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
        self.cell_sectors = self._extract_segment_indices(sector_indices)

    def _compute_sparse_coordinates(
        self,
        rho_max: int,
        eps: float = 1e-8,
    ) -> tp.Tuple[np.ndarray, int]:
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
                The maximal radial offset of a cell relative to the centre of the visual field.

            eps (float):
                A small constant used to prevent division by 0 or taking a log of 0.

        Returns:
            tp.Tuple[np.ndarray, int]:
                A tuple containing:
                    1. The sparse coordinates of the receptive fields.
                    2. The maximal size of the receptive fields.
        """
        # Coordinates of the central pixel
        h2 = self.height // 2
        w2 = self.width // 2

        # Size of the foveal region
        rho_fovea = math.log(rho_max)

        # Number of sectors.
        #
        # NOTE
        # Ideally, this should be divisible by 2 because
        # then the receptive field distribution is symmetric
        # and tends to cover the entire visual field.
        #
        # TODO
        # Figure out if we should actually *enforce* that
        # the sectors be divisible by 2.
        # ==================================================
        S = self.sector_count

        # Sector size.
        q = S / (2 * np.pi)

        # Growth factor for coupling S with R below.
        # This preserves the pixel aspect ratio.
        a = 1 + 1 / q

        # Number of radial rings (coupled with the number of sectors)
        log_a = np.log(a)
        R = np.floor(np.log(rho_max / rho_fovea) / log_a)

        # Cartesian coordinate mesh (x, y)
        rf_rows = np.arange(self.height) - h2
        rf_cols = np.arange(self.width) - w2
        cart_prod = pcf.cartesian_prod(rf_rows, rf_cols)
        (Rs, Cs) = cart_prod[:, 0], cart_prod[:, 1]

        # Logpolar coordinate mesh (r, φ)
        radii = np.sqrt(Rs**2 + Cs**2)
        radial_ratios = Rs / (radii + eps)
        angles = np.arccos(radial_ratios)
        angles = np.where(Cs > 0, 2 * np.pi - angles, angles)

        # Eccentricity-dependent logpolar coordinates of the cells
        # ξ: Radius
        # η: Angle
        ksi = np.log(eps + radii / rho_fovea) / log_a
        eta = q * angles

        # Floored versions of ξ and η (mappable to pixel coordinates)
        u = np.round(ksi)
        v = np.round(eta)

        # Mesh of discrete logpolar coordinates of the RF centres
        # Rs: Radii
        # Ts: Angles
        rf_radii = np.unique(rho_fovea * (a**u))
        rf_angles = np.unique(v / q)

        polar_cart_prod = pcf.cartesian_prod(rf_radii, rf_angles)
        (Rs, Ts) = polar_cart_prod[:, 0], polar_cart_prod[:, 1]

        # Convert Rs and Ts back to integer (x, y) coordinates
        rf_rows = (np.round(np.sin(Ts) * Rs) + h2).astype(np.int32)
        rf_cols = (np.round(np.cos(Ts) * Rs) + w2).astype(np.int32)

        # Create the (x, y) coordinates
        coords = np.stack((rf_rows, rf_cols), axis=1)

        # Prune coordinates that fall outside the image boundaries
        (coords, mask, indices) = self._crop_to_vf(coords)

        # Segment the coordinates into rings and sectors
        self._segment(polar_cart_prod, mask, indices)

        # Compute the maximal RF size
        # ==================================================
        if self.inverse:
            # TODO
            # This needs to be changed.
            # Right now, it's the same as the else branch.
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

        # Compute the coordinates and the RF size ratio
        if self.dense:
            # TODO: Design a proper dense version
            (coords, max_rf_size) = self._compute_sparse_coordinates(rho_max)
        else:
            (coords, max_rf_size) = self._compute_sparse_coordinates(rho_max)

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
        distances = np.sqrt((coords[:, 0] - h2) ** 2 + (coords[:, 1] - w2) ** 2)

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

            rf_size = np.array([rf_size, rf_size])

            # RF spread.
            # The spread is assumed to represent the full size
            # of the RF (e.g., the semi-major axes in the
            # case of an elliptic RF).
            # We have to scale and then halve it to make sure
            # that the RF factories work as expected.
            # ==================================================
            rf_spread = np.max(
                np.vstack((rf_size, self.kernel_params.min_size)), axis=0
            )[0]
            rf_spread *= self.kernel_params.scale * self.kernel_params.aspect

            # print(
            #     f"==[ rf_centre: {rf_centre} | distance: {distance} | rf_size: {rf_size} | rf_spread: {rf_spread}"
            # )

            # Create the RF
            # ==================================================
            result = self._kernel_filter_dispatcher(
                rf_centre,  # Centre of the receptive field
                rf_spread,  # E.g., diameter in the case of a circle
                **self.kernel_params.params,  # Additional RF parameters
            )

            # Sanity check
            if result is None:
                continue

            # Unpack and process the result
            # ==================================================
            (rf_rows, rf_cols, rf_vals, rf_idx) = result

            # Update the rows, columns and values
            rows.append(rf_rows)
            cols.append(rf_cols)
            sparse_rows.append(np.full_like(rf_idx, len(sparse_rows)))
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
        sparse_rows = np.concatenate(sparse_rows)
        sparse_cols = np.concatenate(sparse_cols)
        sparse_vals = np.concatenate(sparse_vals)

        # Create the actual receptive fields
        # ==================================================
        self.forward_synapses = csc_array(
            (
                sparse_vals,
                (sparse_rows, sparse_cols),
            ),
            # Size of the dense tensor, necessary for correct multiplication
            shape=(
                neuron_count,  # rows
                len(self.substrate),  # columns
            ),
            dtype=conf.dtype,
        )

        self.debug(
            f"RF sparsity: {100 * self.forward_synapses.nnz / np.prod(self.forward_synapses.shape):3.4}%"
        )

        # Store various useful arrays and scalars.
        #
        # NOTE
        # These might be redundant, but for now they
        # can be used for visualisation purposes.
        # ==================================================
        self.rf_rows = rows
        self.rf_cols = cols
        self.rf_vals = sparse_vals
        self.cell_coordinates = np.vstack(cell_coordinates)
        self.rf_sizes = np.vstack(rf_sizes)
        self.neuron_count = neuron_count
        self.rf_coords = np.vstack(
            (
                np.concatenate(rows),
                np.concatenate(cols),
            )
        )

        # Compute the RF weight factors.
        # This is necessary for handling overlapping RFs,
        # for instance, in the case of horizontal cells.
        # ==================================================
        if self.create_feedback:
            (unique_rf_coords, occurrences) = np.unique(
                self.rf_coords,
                return_counts=True,
                axis=1,
            )
            rf_dict = {
                tuple(coord): 1 / occ
                for (coord, occ) in zip(unique_rf_coords.T, occurrences)
            }

            # Swap the sparse rows and columns in order to
            # emulate feedback connections.
            # ==================================================
            fb_sparse_rows = np.array(sparse_cols)
            fb_sparse_cols = np.array(sparse_rows)
            fb_sparse_vals = []
            for rf_rows, rf_cols in zip(rows, cols):
                fb_sparse_vals.extend(
                    [rf_dict[(row, col)] for row, col in zip(rf_rows, rf_cols)]
                )

            # Feedback synapses
            self.feedback_synapses = csc_array(
                (
                    fb_sparse_vals,
                    (fb_sparse_rows, fb_sparse_cols),
                ),
                # Size of the dense tensor, necessary for correct multiplication
                shape=(
                    len(self.substrate),  # rows
                    neuron_count,  # columns
                ),
                dtype=conf.dtype,
            )

        self.debug(f"Receptive fields created for {self.neuron_count} neurons.")

    def _make_cartesian_rfs(self):
        """
        TODO: Implement this. :)
        """
        pass

    def make_rfs(self, *args, **kwargs):
        return self._rf_arrangement_dispather(*args, **kwargs)
