from tqdm import tqdm
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse import csr_array
from typing import Any

from collections.abc import Iterable
from collections.abc import Callable
import skimage as ski
from bokeh.plotting import figure

from pyrception import conf
from pyrception import logger
from pyrception import utils
from pyrception.utils import plot
from pyrception.utils.mixins import LoggingMixin
from pyrception.visual.kernel import Kernel
from pyrception.utils.enums import KernelShape
from pyrception.utils.enums import KernelFilter
from pyrception.utils.enums import RFArrangement


class ReceptiveFields(LoggingMixin):

    def __init__(
        self,
        size: tuple[int, ...] | np.ndarray,
        sectors: int = 64,
        arrangement: RFArrangement | str = RFArrangement.LogPolar,
        phyllotactic: bool = False,
        ksize: int | tuple[int, ...] | np.ndarray = None,
        kscale: tuple[int, ...] | np.ndarray | None = None,
        kbounds: tuple[int, ...] | np.ndarray | None = None,
        kangle: float = 0.0,
        kshape: KernelShape | str = KernelShape.Elliptic,
        kfilter: KernelFilter | str = KernelFilter.Uniform,
        kparams: dict[str, Any] | None = None,
        substrate: np.ndarray | None = None,
        name: str | None = None,
        notifier: Callable | None = None,
    ):
        """
        A set of receptive fields that can be used as the base for a layer.

        Args:
            size:
                The dimensions of the visual field (height, width, depth).
                NOTE: Colour vision is not implemented yet.
                It would be necessary to take into account the depth dimension.

            sectors:
                Number of sectors ('wedges') for logpolar receptive fields.

            arrangement:
                Defines how the RFs are arranged spatially to cover the visual field.

            phyllotactic:
                Switch for phyllotactic arrangement of receptive fields. Only relevant for logpolar arrangement.

            ksize:
                The kernel size. Only used for Cartesian arrangements at the moment.
                For logpolar arrangements, the kernel size is computed based on the distance from the fovea.

            kscale:
                A scaling factor for kernels.
                A 2D value indicates that each dimension should be scaled independently.
                Larger values result in kernels that may overlap more, while smaller
                values may result in kernels that leave gaps in the visual field.

            kbounds:
                Kernel size bounds. Can be specified in any of the following forms:
                    - (min_x, min_y, max_x, max_y): Upper and lower bounds specified.
                    - (min_x, min_y): Implicitly no upper bound.
                    - (1, 1, max_x, max_y): No lower bound (beyond the trivial (1, 1)).

            kangle:
                Kernel rotation angle in degrees. Assumed to increase from 0 to 360 *counterclockwise*.

            kshape:
                The kernel shape.
                NOTE: This is *not* the same as a tensor shape.
                Rather, it is the geometric shape of the kernel (rectangular, elliptic, etc.).

            kfilter:
                The filter response type for receptive fields in this layer.

            kparams:
                Extra parameters to pass to the kernel factory function.

            substrate:
                The coordinates of the input cells that constitute the 'substrate'
                to which the receptive fields are applied.

            name:
                An optional name for this instance.

            notifier:
                A progress notification function.
        """

        super().__init__(name, notifier)

        size = utils.arg2np(size, ext=3)
        self.size = size[:2]
        self.sectors = int(sectors)
        self.arrangement = RFArrangement(arrangement)
        self.phyllotactic = phyllotactic
        self.ksize = utils.arg2np(
            ksize or (3, 3),
            ext=2,
            fill=True,
            bounds=(1, None),
        )
        self.kscale = utils.arg2np(
            kscale or (1.0, 1.0),
            bounds=(0.05, None),
            dtype=np.float32,
            pad=2,
            val=kscale,
            fill=True,
            ext=2,
        )
        self.kbounds = utils.arg2np(
            kbounds or (1, 1),
            ext=4,
            pad=4,
            val=np.iinfo(np.int32).max,
            fill=True,
        )
        self.kangle = float(kangle)
        self.kshape = KernelShape(kshape)
        self.kfilter = KernelFilter(kfilter)
        self.kparams = kparams or {}
        self.substrate = (
            utils.make_substrate(self.size[0], self.size[1])
            if substrate is None
            else substrate
        )

        # Internal attributes used for constructing receptive fields.
        # ==================================================
        self.height = self.size[0].item()
        self.width = self.size[1].item()
        self.depth = 1 if size.size < 3 else size[2].item()
        self.center = self.size // 2
        self.cell_count: int = 0
        self.cell_coordinates: np.ndarray | None = None
        self.cell_indices: tuple[np.ndarray, ...] | None = None
        self.forward_synapses: np.ndarray | None = None
        self.feedback_synapses: np.ndarray | None = None
        self.rf_indices: np.ndarray | None = None
        self.rf_rows: np.ndarray | None = None
        self.rf_cols: np.ndarray | None = None
        self.rf_vals: np.ndarray | None = None
        self.rf_sizes: np.ndarray | None = None
        self.fb_counts: np.ndarray | None = None
        self.cell_rows: np.ndarray | None = None
        self.cell_cols: np.ndarray | None = None
        self.cell_rings: np.ndarray | None = None
        self.cell_sectors: np.ndarray | None = None
        self.kernels: dict | None = None

        self.make_rfs()

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls} | size: {self.size} | cell count: {self.cell_count})"

    def _flatten(
        self,
        container: Iterable,
    ) -> np.ndarray:
        """
        Flatten a potentially nested (heterogeneous) array of elements.

        TODO: Move this into the `utils` module.

        Args:
            container: A container or a numeric value.

        Returns:
            A NumPy array.
        """
        if isinstance(container, (int, float)):
            container = [container]

        elif isinstance(container, (list, tuple, set, np.ndarray)):
            arr = []
            for sub in container:
                arr.extend(self._flatten(sub))

            container = arr

        return np.unique(container).tolist()

    def _make_arrangement(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Make a cell arrangement according to the `arrangement` attribute.

        Raises:
            Raised if the requested RF arrangement is invalid.

        Returns:
            The resulting cell arrangement.
        """

        match self.arrangement:
            case RFArrangement.LogPolar:
                return self._make_logpolar_arrangement()

            case RFArrangement.Cartesian:
                return self._make_cartesian_arrangement()

            case _:
                raise TypeError(f"Invalid arrangement '{self.arrangement}'")

    def _make_logpolar_arrangement(self) -> tuple[np.ndarray, ...]:
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
            The coordinates of the cells.
        """

        self.logger.debug("Creating logpolar grid...")

        # Coordinates of the central pixel
        h2 = self.center[0]
        w2 = self.center[1]

        # Sectors
        S = self.sectors

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
        polar_coords = utils.cartesian_prod(radii, angles)

        if self.phyllotactic:
            # Stagger the radial component by 0.5 every other sector
            ridx = np.arange(R + 1)
            offset = (np.arange(S + 1) % 2) / 2
            polar_coords[:, 0] += utils.cartesian_prod(ridx, offset)[:, 1]

        polar_coords[:, 0] = q * (a ** polar_coords[:, 0])

        fovea_radii = np.arange(q)
        polar_coords = np.concatenate(
            (utils.cartesian_prod(fovea_radii, angles), polar_coords)
        )

        rf_rows = np.round(np.sin(polar_coords[:, 1]) * polar_coords[:, 0] + h2)
        rf_cols = np.round(np.cos(polar_coords[:, 1]) * polar_coords[:, 0] + w2)

        # Create the (x, y) coordinates
        cell_coords = np.stack((rf_rows, rf_cols), axis=1).astype(np.int32)

        # Crop the coordinates to the FoV
        cropped, mask, indices = utils.crop_to_fov(cell_coords, self.size)
        cell_coords = cropped[indices]
        polar_coords = polar_coords[mask][indices]

        # Segment the coordinates into rings and sectors.
        # First index the polar coordinates and then sort them
        # by radius and by angle.
        index = np.arange(polar_coords.shape[0])[:, None]
        radius_angle = np.concatenate((polar_coords, index), axis=1)
        angle_radius = np.concatenate((np.roll(polar_coords, 1, axis=1), index), axis=1)
        sorted_by_radius = np.array(sorted([tuple(c) for c in radius_angle.tolist()]))
        sorted_by_angle = np.array(sorted([tuple(c) for c in angle_radius.tolist()]))

        self.cell_rings = self._extract_segment_indices(sorted_by_radius)
        self.cell_sectors = self._extract_segment_indices(sorted_by_angle)

        kernel_sizes = cell_coords - self.center
        kernel_sizes = np.sqrt((kernel_sizes**2).sum(axis=1))
        kernel_sizes = 2 * kernel_sizes / (3 * q) + a

        kernel_sizes = np.clip(
            self.kscale
            * np.stack(
                (
                    kernel_sizes,
                    kernel_sizes,
                ),
                axis=1,
            ),
            a_min=self.kbounds[:2],
            a_max=self.kbounds[2:4],
        )

        return (cell_coords, kernel_sizes)

    def _make_cartesian_arrangement(self):
        """
        Regular Cartesian arrangement.

        Returns:
            The coordinates of the cells.
        """

        self.logger.debug("Creating a Cartesian grid...")

        # Compute the stride from the kernel
        stride = np.clip(self.ksize // 2 + 1, 1, None)

        # Row and column grid
        rf_rows = np.arange(0, self.height, stride[0])
        rf_cols = np.arange(0, self.width, stride[1])

        # Create the cartesian product as (x, y) coordinates
        cell_coords = utils.cartesian_prod(rf_rows, rf_cols)

        # Crop the coordinates to the FoV
        cropped, mask, indices = utils.crop_to_fov(cell_coords, self.size)
        cell_coords = cropped[indices]

        kernel_sizes = np.copy(self.ksize)

        if len(kernel_sizes.shape) == 1:
            # Add a dimension if there is only one
            kernel_sizes = kernel_sizes[None, :]

        kernel_sizes = np.repeat(kernel_sizes, len(cell_coords), axis=0)

        return (cell_coords, kernel_sizes)

    def _extract_segment_indices(
        self,
        container: Iterable,
    ) -> list[np.ndarray]:
        """
        Extract the indices of a specific segment.
        For now, this is limited to rings and sectors,
        but it should work with any type of segment
        (e.g., row, column, kernel...) as long as the
        coordinate indices are sorted appropriately.

        Args:
            container:
                An iterable containing tuples of some metric
                (e.g., radius, angle, etc.) and its index.
                The container should be sorted in such a way
                that each segment is represented by a continuous
                portion of the container.

        Returns:
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

    def make_rfs(self, *args, **kwargs):
        (cell_coords, kernel_sizes) = self._make_arrangement(*args, **kwargs)

        unique_ksizes = np.unique(kernel_sizes, axis=0)

        # Proto-kernels
        proto_k_coords = {}
        proto_k_outlines = {}
        proto_k_weights = {}
        for ksize in unique_ksizes:
            key = tuple(ksize.tolist())
            _coords, _outlines = Kernel.make_kernel(
                self.kshape,
                ksize,
                self.kangle,
            )
            _weights = Kernel.make_filter(
                self.kfilter,
                _coords[0],
                _coords[1],
                angle=self.kangle,
                **self.kparams,
            )
            proto_k_coords[key] = _coords
            proto_k_outlines[key] = _outlines
            proto_k_weights[key] = _weights

        rf_coords = [None for _ in range(len(kernel_sizes))]
        sp_vals = [None for _ in range(len(kernel_sizes))]
        sp_row_idx = [None for _ in range(len(kernel_sizes))]
        sp_col_idx = [None for _ in range(len(kernel_sizes))]
        outline_coords = [None for _ in range(len(kernel_sizes))]
        kspan = np.zeros((2,), dtype=np.int32)

        kernels = {}
        kernel_range = enumerate(kernel_sizes)

        if self.verbose():
            kernel_range = tqdm(
                kernel_range, desc="Building kernels...", total=len(cell_coords)
            )

        # The index map is a 2D array where the value of each
        # substrate cell is its index in the real substrate array.
        # By indexing the substrate below when creating a kernel,
        # it is possible to extract the sparse (row) indices for the
        # kernel very efficiently.
        substrate_idx = np.zeros(self.size, dtype=np.int32)
        substrate_idx[self.substrate[:, 0], self.substrate[:, 1]] = np.arange(
            1, len(self.substrate) + 1, dtype=np.int32
        )
        weight_idx = np.zeros(self.size)

        # The offset represents the beginning of the kernel's receptive field
        # relative to the beginning of the unrolled substrate.
        offset = 0
        for kidx, ksize in kernel_range:
            key = tuple(ksize.tolist())
            _k_coords = np.column_stack(proto_k_coords[key]) + cell_coords[kidx]
            _k_outline = np.column_stack(proto_k_outlines[key]) + cell_coords[kidx]

            cropped_kernel, kernel_mask, _ = utils.crop_to_fov(_k_coords, self.size)
            weight_idx[cropped_kernel[:, 0], cropped_kernel[:, 1]] = proto_k_weights[
                key
            ][kernel_mask]

            # Overlap with the substrate
            overlap = substrate_idx[cropped_kernel[:, 0], cropped_kernel[:, 1]]
            coverage_idx = overlap != 0
            coverage = overlap[coverage_idx] - 1

            # Re-extract the coordinates from the indices and the substrate.
            # This is necessary to sparsify the kernel coordinates for layers with sparse substrates.
            _k_coords = self.substrate[coverage]

            weights = np.copy(weight_idx[_k_coords[:, 0], _k_coords[:, 1]])
            weights /= weights.sum()
            weight_idx[:] = 0
            cropped_outline, _, _ = utils.crop_to_fov(_k_outline, self.size)

            kspan[0] = kspan[1].item()
            kspan[1] += len(weights)

            rf_coords[kidx] = _k_coords
            sp_vals[kidx] = weights
            outline_coords[kidx] = cropped_outline

            kernel = Kernel(
                ksize,
                cell_coords[kidx],
                self.kshape,
                self.kfilter,
                self.kangle,
                index=kidx,
                coords=_k_coords,
                outline=cropped_outline,
                weights=weights,
                span=np.copy(kspan),
                offset=np.arange(offset, offset + len(weights)),
            )

            sp_row_idx[kidx] = np.full((len(kernel.coords),), kidx, dtype=np.int32)
            sp_col_idx[kidx] = coverage
            kernels[kidx] = kernel
            offset += len(weights)

        rf_coords = np.concatenate(rf_coords)
        sp_rows = np.concatenate(sp_row_idx, dtype=np.int32)
        sp_cols = np.concatenate(sp_col_idx, dtype=np.int32)

        self.cell_coordinates = cell_coords
        self.kernels = kernels
        self.cell_count = len(cell_coords)
        self.rf_rows = rf_coords[:, 0]
        self.rf_cols = rf_coords[:, 1]
        self.rf_vals = np.concatenate(sp_vals, dtype=np.float32)

        # Forward synapses.
        # ==================================================
        self.forward_synapses = csr_array(
            (self.rf_vals, (sp_rows, sp_cols)),
            shape=(self.cell_count, len(self.substrate)),
            dtype=np.float32,
        )

        self.logger.debug(
            f"Sparsity: {100 * self.forward_synapses.nnz / np.prod(self.forward_synapses.shape):3.4}%"
        )

        # Feedback synapses send a regulating
        # signal back to the input substrate.
        # ==================================================
        self.logger.debug("Creating feedback synapses.")
        (unique, inverse, counts) = np.unique(
            rf_coords,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        self.fb_counts = counts[inverse]

        # NOTE: Rows and columns are swapped here.
        self.feedback_synapses = csc_array(
            (1 / self.fb_counts, (sp_cols, sp_rows)),
            shape=(len(self.substrate), self.cell_count),
            dtype=np.float32,
        )

        self.logger.debug(f"Created {self.cell_count} cells.")

        return kernels

    def visualise(
        self,
        cells: int | list[int] | None = None,
        cell_colour: tuple[str, tuple[int, ...]] | None = "#00ffffff",
        kernel_colour: tuple[str, tuple[int, ...]] | None = "#ffffffff",
        outline_colour: tuple[str, tuple[int, ...]] | None = "#ffff0077",
        weighted: bool = False,
        title: str | None = None,
    ) -> figure:
        """
        Visualise the cells and their receptive fields.

        TODO: This is a temporary workaround until excitatory / inhibitory
        substrates are moved into the ReceptiveFields class so that they
        don't have to be passed as arguments to self.visualise().

        Args:

            cells: Coordinates of the cells to plot.
            cell_colour: The colour to use for highlighting the plotted cells.
            kernel_colour: The colour to use for highlighting the plotted receptive field.
            outline_colour: The colour to use for drawing the kernel outlines.
            weighted: If set, use the kernel weights, otherwise use a uniform value.

        Returns:
            A visualisation of the receptive fields.
        """

        # Bipolar cell input for a single amacrine cell
        canvas = np.zeros((self.height, self.width, 4))

        # Convert HEX colours to RGBA
        if cell_colour is not None:
            cell_colour = utils.to_rgba(cell_colour)
        if kernel_colour is not None:
            kernel_colour = utils.to_rgba(kernel_colour)[None, :]
        if outline_colour is not None:
            outline_colour = utils.to_rgba(outline_colour)[None, :]

        # Process the requested cell coordinates.
        cells = (
            np.arange(len(self.cell_coordinates), dtype=np.uint32)
            if cells is None
            else self._flatten(cells)
        )

        kernels = [self.kernels[k] for k in cells]
        rf_coords = np.concatenate([k.coords for k in kernels], axis=0)
        centers = np.concatenate([k.center[None, :] for k in kernels], axis=0)
        offsets = np.concatenate([k.offset for k in kernels])

        if kernel_colour is not None:
            # Plot the receptive fields
            if weighted:
                kernel_colour = (
                    kernel_colour
                    * np.concatenate([k.weights for k in kernels])[:, None]
                )

            # Use logarithmic scaling for the counts, otherwise the intensity
            # fades too quickly away from the fovea.
            kernel_colour = kernel_colour * np.log1p(self.fb_counts[offsets][:, None])

            canvas[
                rf_coords[:, 0],
                rf_coords[:, 1],
            ] = kernel_colour
            canvas = ski.exposure.rescale_intensity(canvas, out_range=(0.0, 1.0))

        if cell_colour is not None:
            # Plot the cells last so that they are superimposed onto the receptive fields.
            canvas[
                centers[:, 0],
                centers[:, 1],
            ] = cell_colour

        if outline_colour is not None:
            # Potentially plot the outlines
            outlines = np.concatenate([k.outline for k in kernels], axis=0)

            # Plot the cells last so that they are superimposed onto the receptive fields.
            canvas[
                outlines[:, 0],
                outlines[:, 1],
            ] = outline_colour

        return plot.show_composite(plot.image(canvas), title=title)
