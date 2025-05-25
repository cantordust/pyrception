from tqdm import tqdm
import numpy as np
from scipy.sparse import csc_array

import typing as tp

from pyrception import conf
from pyrception.utils.types import KernelShape
from pyrception.utils.types import KernelFilter
from pyrception.utils.types import RFArrangement
from pyrception.utils.logging import LoggingMixin
from pyrception.visual.kernel import Kernel
import pyrception.utils.functions as pcf


class ReceptiveFields(LoggingMixin):
    """
    A set of receptive fields for a specific substrate.
    """

    def __init__(
        self,
        size: tuple[int, ...],
        substrate: np.ndarray = None,
        sectors: int = 64,
        arrangement: RFArrangement = RFArrangement.LogPolar,
        shape: KernelShape = KernelShape.Elliptic,
        filter: KernelFilter = KernelFilter.Uniform,
        extent: np.ndarray = np.array([1.0, 1.0]),
        scale: np.ndarray = np.array([1.0, 1.0]),
        min_size: np.ndarray = np.array([1.0, 1.0]),
        angle: float = 0.0,
        dense: bool = False,
        feedback: bool = False,
        phyllotactic: bool = False,
        name: str = "Receptive fields",
        kernel_params: dict[str, tp.Any] = None,
        notifier: tp.Callable = None,
    ):
        """

        Args:
            size (tuple[int, ...]):
                The dimensions of the visual field (height, width, depth).
                NOTE: Colour vision is not implemented yet.
                It would be necessary to take into account the depth dimension.

            substrate (np.ndarray):
                The coordinates of the input cells that constitute the 'substrate' to which
                the receptive fields are applied. These coordinates could be sparse.

            sectors (int, optional):
                Number of sectors ('wedges') for logpolar receptive fields. Defaults to 64.

            arrangement (RFArrangement, optional):
                Defines how the RFs are arranged spatially to cover the visual field.
                Defaults to RFArrangement.LogPolar.

            shape (KernelShape, optional):
                The shape of the kernel.
                NOTE: This is *not* the same as a tensor shape - rather,
                it is the geometric (2D) shape of the kernel.
                Defaults to KernelShape.Elliptic.

            filter (KernelFilter, optional):
                The filter response type for receptive fields in this layer.
                Defaults to KernelFilter.Uniform.

            extent (np.ndarray, optional):
                Extent of the receptive field coverage.
                A value of 1.0 means that the entire visual field is covered.
                Defaults to array([1, 1]).

            scale (np.ndarray, optional):
                A scaling factor for kernels.
                Larger values result in kernels that may overlap more,
                while smaller values may result in kernels that leave gaps in
                the visual field.
                Defaults to array([1, 1]).

            min_size (np.ndarray, optional):
                Minimal kernel size (usually restricted to the foveal region).
                Defaults to array([1, 1]).

            angle (float, optional):
                Angle of the receptive field. Should be between -np.pi and np.pi.
                Defaults to 0.0.

            sparse (bool, optional):
                Indicate that the receptive fields should be sparse.
                This is relevant for the amacrine and ganglion layers.
                Defaults to False.

            feedback (bool, optional):
                Switch for receptive field feedback connections.
                The factors are used to average out the feedback of overlapping receptive fields.
                Mainly used for (inhibitory) feedback neurons, such as horizontal and amacrine cells.
                Defaults to False.

            phyllotactic (bool, optional):
                Switch for phyllotactic arrangement of receptive fields. Only relevant for logpolar arrangement.
                Defaults to False.

            name (str, optional):
                Layer name. Defaults to "Receptive fields".
                Each derived class overrides the name with a default value.

            kernel_params (dict[str, tp.Any], optional):
                Extra parameters to pass to the kernel factory function.
                Defaults to None.

            notifier (tp.Callable):
                A progress notification function
        """

        # Initialise the base
        super().__init__(name, notifier)

        if kernel_params is None:
            kernel_params = {}

        # Various dimensions
        # TODO: add some assertions
        # ==================================================
        size = np.array(size, dtype=np.int32)
        self.size = size[:2]
        self.height = int(size[0])
        self.width = int(size[1])
        self.depth = 1 if len(size) == 2 else int(size[2])  # NOTE: Unused for now.
        self.sector_count = sectors
        if isinstance(extent, (int, float)):
            extent = [extent, extent]
        self.extent = np.array(extent)
        self.fovea_size = int(0.5 * self.sector_count / np.pi)
        self.crop = np.array(((1 - self.extent) * self.size) / 2, dtype=np.int32)
        self.fov = self.size - 2 * self.crop
        self.center = self.size // 2
        self.crop_center = self.center - self.crop
        self.top = self.crop[0]
        self.bottom = self.height - self.crop[0]
        self.left = self.crop[1]
        self.right = self.width - self.crop[1]
        self.arrangement = arrangement
        self.shape = shape
        self.filter = filter
        if isinstance(scale, (int, float)):
            scale = [scale, scale]
        self.scale = np.clip(np.array(scale, dtype=np.float32), 0.0, None)
        if isinstance(min_size, (int, float)):
            min_size = [min_size, min_size]
        self.min_size = np.clip(np.array(min_size, dtype=np.int32), 1, None)
        self.angle = np.mod(angle, 2 * np.pi) - np.pi
        self.dense = dense
        self.feedback = feedback
        self.phyllotactic = phyllotactic
        self.kernel_params = {} if kernel_params is None else kernel_params

        # Internal parameters used for constructing RFs.
        # ==================================================
        self.neuron_count = 0
        self.substrate = self._make_substrate() if substrate is None else substrate
        self.forward_synapses = None
        self.feedback_synapses = None
        self.rf_indices = None
        self.rf_rows = None
        self.rf_cols = None
        self.rf_vals = None
        self.rf_sizes = None
        self.cell_rows = None
        self.cell_cols = None
        self.cell_rings = None
        self.cell_sectors = None
        self.kernels = None

    @property
    def _rf_arrangement_dispatcher(self) -> tp.Callable:
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
            RFArrangement.LogPolar: self._make_logpolar_distribution,
            RFArrangement.Cartesian: self._make_cartesian_distribution,
        }

        function = functions.get(self.arrangement, None)

        if function is None:
            raise TypeError(f"Invalid RF arrangement '{self.arrangement}'")

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

    def _crop(
        self,
        cartesian: np.ndarray,
        polar: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """
        Trim the coordinates of the receptive fields to the
        dimensions of the visual field.

        Args:
            cartesian (np.ndarray):
                Pixel coordinates of the receptive fields in Cartesian coordinates.

            polar (np.ndarray):
                Pixel coordinates of the receptive fields in polar coordinates.

        Returns:
            tuple[np.ndarray, ...]:
                A tuple containing:
                    1. The trimmed coordinates.
                    2. The mask that trims the coordinates to the visual field.
                    3. An index array for the subset of unique coordinates
                        (since some coordinates might be repeated).
        """

        cartesian = cartesian.astype(np.int32)

        # Mask to trim the indices to the visual field.
        fov_mask = (
            (cartesian[:, 0] >= self.top)
            & (cartesian[:, 0] < self.bottom)
            & (cartesian[:, 1] >= self.left)
            & (cartesian[:, 1] < self.right)
        )

        # Return unique indices.
        # ==================================================
        cartesian = cartesian[fov_mask]
        polar = polar[fov_mask]

        unique_indices = np.unique(
            cartesian,
            return_index=True,
            axis=0,
        )[1].astype(
            dtype=np.int32,
        )

        return (cartesian[unique_indices], polar[unique_indices])

    def _extract_segment_indices(
        self,
        container: tp.Iterable,
    ) -> list[np.ndarray]:
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
            list[np.ndarray]:
                The indices of the coordinates of the segment.
        """

        segments = []
        segment = []
        cur_item = -1.0
        for _, item, index in container:
            if item < cur_item:
                segments.append(np.array(segment, dtype=np.int32))
                segment = [index]
                cur_item = -1.0
            else:
                segment.append(index)
                cur_item = item

        # The last segment might be incomplete
        if len(segment) > 0:
            segments.append(np.array(segment, dtype=np.int32))

        return segments

    def _segment(
        self,
        polar: np.ndarray,
    ):
        # Index the polar coordinates, and then
        # sort them by radius and by angle
        index = np.arange(polar.shape[0])[:, None]
        radius_angle = np.concatenate((polar, index), axis=1)
        angle_radius = np.concatenate((np.roll(polar, 1, axis=1), index), axis=1)
        sorted_by_radius = np.array(sorted([tuple(c) for c in radius_angle.tolist()]))
        sorted_by_angle = np.array(sorted([tuple(c) for c in angle_radius.tolist()]))

        # Extract the ring coordinates.
        # ==================================================
        self.cell_rings = self._extract_segment_indices(sorted_by_radius)

        # Extract the sector coordinates.
        # ==================================================
        self.cell_sectors = self._extract_segment_indices(sorted_by_angle)

    def _make_logpolar_distribution(self) -> tuple[np.ndarray, ...]:
        """
        Implementation of eccentricity-dependent log-polar distribution
        of receptive fields as outlined in the following papers,
        with slight corrections and improvements.

        References:
            [1] Maiello, G., Chessa, M., Bex, P. J. & Solari, F.
                Near-optimal combination of disparity across a log-polar scaled visual field.
                PLoS Comput Biol 16, e1007699 (2020).

            [2] Araujo, H. and Dias, J.M. (1997)
                An introduction to the log-polar mapping,
                Proceedings II Workshop on Cybernetic Vision.

        Returns:
            np.ndarray:
                The coordinates of the cells.
        """

        self.info("Creating logpolar receptive fields...")

        # Coordinates of the central pixel
        h2 = self.center[0]
        w2 = self.center[1]

        # Sectors
        S = self.sector_count

        # Sector size, which is also the size of the fovea.
        q = 0.5 * S / np.pi

        # Maximal offset of the centre of the log-polar
        # rings from the centre of the FoV.
        max_fov = (np.sqrt(h2**2 + w2**2)).astype(np.int32)

        # Growth factor for coupling S with R below.
        # This preserves the pixel aspect ratio.
        a = 1 + 1 / q

        # Number of radial rings (coupled with the number of sectors)
        R = np.round(np.log(max_fov / q) / np.log(a)).astype(np.int32)

        radii = np.arange(R + 1)
        angles = np.linspace(0, 2 * np.pi, S + 1)
        polar_coords = pcf.cartesian_prod(radii, angles)

        if self.phyllotactic:
            # Stagger the radial component by 0.5 every other sector
            ridx = np.arange(R + 1)
            offset = (np.arange(S + 1) % 2) / 2
            polar_coords[:, 0] += pcf.cartesian_prod(ridx, offset)[:, 1]

        polar_coords[:, 0] = q * (a ** polar_coords[:, 0])

        fovea_radii = np.arange(q)
        polar_coords = np.concatenate(
            (pcf.cartesian_prod(fovea_radii, angles), polar_coords)
        )

        rf_rows = np.round(np.sin(polar_coords[:, 1]) * polar_coords[:, 0] + h2)
        rf_cols = np.round(np.cos(polar_coords[:, 1]) * polar_coords[:, 0] + w2)

        # Create the (x, y) coordinates
        cartesian_coords = np.stack((rf_rows, rf_cols), axis=1).astype(np.int32)

        # Crop the coordinates to the FoV
        (cartesian_coords, polar_coords) = self._crop(cartesian_coords, polar_coords)

        # Segment the coordinates into rings and sectors
        self._segment(polar_coords)

        rf_coords = cartesian_coords - self.center
        rf_coords = np.ceil(np.sqrt((rf_coords**2).sum(axis=1)))
        receptive_field_sizes = 2 * (rf_coords / (3 * q) + a)

        return (cartesian_coords, polar_coords, receptive_field_sizes)

    def _make_cartesian_distribution(self):
        """
        TODO: Implement this.
        """
        pass

    def make_rfs(self, *args, **kwargs):
        (cartesian_coords, polar_coords, receptive_field_sizes) = (
            self._rf_arrangement_dispatcher(*args, **kwargs)
        )

        receptive_field_sizes = np.clip(
            self.scale
            * np.stack(
                (
                    receptive_field_sizes,
                    receptive_field_sizes,
                ),
                axis=1,
            ),
            self.min_size,
            None,
        )

        # The index map is a 2D array where the value of each
        # input cell is its index in the real substrate array.
        # By indexing the substrate below when creating a kernel,
        # it is possible to extract the sparse (row) indices for the
        # kernel very efficiently.
        index_map = np.zeros(self.size, dtype=np.int32)
        index_map[self.substrate[:, 0], self.substrate[:, 1]] = np.arange(
            len(self.substrate), dtype=np.int32
        )

        self.cell_coordinates = []
        self.kernels = []
        pbar = tqdm(desc="Building kernels...", total=len(cartesian_coords))
        for size, center in zip(receptive_field_sizes, cartesian_coords):
            kernel = Kernel(
                size,
                center,
                self.fov,
                self.crop,
                self.shape,
                self.filter,
                self.angle,
                index_map,
                self.substrate,
                self.kernel_params,
            )
            if kernel.coordinates.size > 0:
                self.cell_coordinates.append(center)
                self.kernels.append(kernel)
                pbar.update()

        self.cell_coordinates = np.array(self.cell_coordinates)
        self.neuron_count = len(self.kernels)

        # Create the receptive fields.
        # ==================================================
        sparse_rows = np.concatenate(
            [np.full((len(k.indices),), idx) for idx, k in enumerate(self.kernels)]
        )
        sparse_cols = np.concatenate([k.indices for k in self.kernels])
        sparse_vals = np.concatenate([k.weights for k in self.kernels])
        self.rf_indices = np.vstack((sparse_rows, sparse_cols))
        self.rf_coords = np.concatenate([k.coordinates for k in self.kernels], axis=0)
        # print(f"==[ coords: {self.rf_coords}")

        self.forward_synapses = csc_array(
            (
                sparse_vals,
                (sparse_rows, sparse_cols),
            ),
            # Shape of the dense tensor, necessary for correct multiplication
            shape=(
                self.neuron_count,  # rows
                len(self.substrate),  # columns
            ),
            dtype=conf.num.numpy.dtype,
        )

        self.debug(
            f"RF sparsity: {100 * self.forward_synapses.nnz / np.prod(self.forward_synapses.shape):3.4}%"
        )

        # Feedback synapses.
        # This is necessary for handling overlapping RFs,
        # for instance, in the case of horizontal cells.
        # ==================================================
        if self.feedback:
            self.debug("Creating feedback synapses.")
            (unique, counts) = np.unique(
                self.rf_coords,
                return_counts=True,
                axis=0,
            )

            rf_dict = {
                tuple(coord): 1 / count for (coord, count) in zip(unique, counts.T)
            }

            fb_vals = np.array([rf_dict[tuple(coord)] for coord in self.rf_coords])

            # Swap the sparse rows and columns in order to
            # emulate feedback connections.
            # ==================================================
            # Feedback synapses.
            self.feedback_synapses = csc_array(
                (
                    fb_vals,
                    (sparse_cols, sparse_rows),
                ),
                # Shape of the dense tensor, necessary for correct multiplication
                shape=(
                    len(self.substrate),  # rows
                    self.neuron_count,  # columns
                ),
                dtype=conf.num.numpy.dtype,
            )

        self.debug(f"Receptive fields created for {self.neuron_count} neurons.")

        return (cartesian_coords, polar_coords, receptive_field_sizes)
