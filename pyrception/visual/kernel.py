import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
from pyrception.utils.logging import Logger
from pyrception.visual.utils.types import KernelFilter
from pyrception.visual.utils.types import KernelShape


class Kernel(Logger):

    def __init__(
        self,
        size: np.ndarray,
        center: np.ndarray,
        fov: np.ndarray = None,
        crop: np.ndarray = None,
        shape: KernelShape = KernelShape.Elliptic,
        filter: KernelFilter = KernelFilter.Uniform,
        angle: float = 0.0,
        substrate: np.ndarray = None,
        params: tp.Dict = None,
        weights: np.ndarray = None,
    ):

        super().__init__("Kernel")
        if params is None:
            params = {}

        if crop is None:
            np.array([0, 0], dtype=np.int32)
        self.crop = crop

        self.size = size
        self.center = center
        if fov is None:
            fov = np.array(size)
        self.fov = fov
        self.shape = shape
        self.filter = filter
        self.angle = angle
        self.params = params
        self.params = params

        self.coordinates = None
        self.indices = None
        self.outline = None
        self.weights = weights

        self.make_shape()
        self.extract_indices(substrate)
        self.make_filter()

    @property
    def make_shape(self):
        """
        Return the kernel factory function based on the kernel shape.

        Raises:
            TypeError:
                Raised if the requested kernel shape is invalid.

        Returns:
            tp.Callable:
                The kernel factory function for the requested kernel shape.
        """

        shape_functions = {
            KernelShape.Elliptic: self._make_elliptic_kernel,
            KernelShape.Rectangular: self._make_rectangular_kernel,
        }

        shape_function = shape_functions.get(self.shape, None)

        if shape_function is None:
            raise TypeError(f"Invalid kernel shape '{self.shape}'")

        return shape_function

    @property
    def make_filter(self):
        """
        Return the kernel factory function based on the kernel filter.

        Raises:
            TypeError:
                Raised if the requested kernel filter is invalid.

        Returns:
            tp.Callable:
                The kernel factory function for the requested kernel filter.
        """

        filter_functions = {
            KernelFilter.Uniform: self._make_uniform_kernel,
            KernelFilter.Gaussian: self._make_gaussian_kernel,
            KernelFilter.Gabor: self._make_gabor_kernel,
        }

        filter_function = filter_functions.get(self.filter, None)

        if filter_function is None:
            raise TypeError(f"Invalid kernel filter '{self.filter}'")

        return filter_function

    def extract_indices(
        self,
        substrate: np.ndarray,
    ):
        """
        Extract the sparse indices of the kernel based on the
        arrangement of input cells in the substrate.

        Args:
            canvas (np.ndarray):
                A 2D array containing the indices of each cell in the substrate.
        """

        # Indices
        # ==================================================
        overlap = substrate[self.coordinates[:, 0], self.coordinates[:, 1]]
        self.indices = overlap[overlap != 0]

    def _make_elliptic_kernel(self):
        """
        Create an elliptic kernel centred at a certain pixel and
        having the specified spread (semi-major axes).
        The kernel can be optionally rotated at an angle.
        """

        center = self.center - self.crop
        size = self.size.astype(np.int32)

        # Coordinates
        # ==================================================
        self.coordinates = (
            np.stack(
                ski.draw.ellipse(
                    center[0],
                    center[1],
                    self.size[0],
                    self.size[1],
                    shape=self.fov,
                    rotation=self.angle,
                ),
                axis=1,
            )
            + self.crop
        )

        # Outline
        # ==================================================
        self.outline = (
            np.stack(
                ski.draw.ellipse_perimeter(
                    center[0],
                    center[1],
                    size[0],
                    size[1],
                    shape=self.fov,
                    orientation=self.angle,
                ),
                axis=1,
            )
            + self.crop
        )

    def _make_rectangular_kernel(self):
        """
        Create a rectangular kernel centred at a certain pixel and
        having the specified spread (side lengths).
        The kernel can be optionally rotated at an angle.

        WIP: This is a stub - to be implemented.
        """

        center = self.center - self.crop
        half_size = self.size.astype(np.int32) // 2

        # Coordinates
        # ==================================================
        self.coordinates = (
            np.stack(
                ski.draw.rectangle(
                    center - half_size,
                    center + half_size,
                    shape=self.fov,
                ),
                axis=1,
            )
            + self.crop
        )

        # Outline
        # ==================================================
        self.outline = (
            np.stack(
                ski.draw.rectangle(
                    center - half_size,
                    center + half_size,
                    shape=self.fov,
                ),
                axis=1,
            )
            + self.crop
        )

    def _make_uniform_kernel(self):
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

        # if substrate is None:
        #     substrate = self.substrate

        # # Get the rows and columns for the kernel
        # (rows, cols, idx, _, _) = self.make(
        #     centre,
        #     spread,
        #     angle,
        #     substrate,
        # )

        # # Bail out if the kernel size is 0
        # if cols.size == 0:
        #     return

        # if len(self.coordinates) == 0:
        #     print(f"==[ center: {self.center}")
        #     print(f"==[ size: {self.size}")

        if self.weights is None:
            self.weights = 1 / self.coordinates.size

        self.weights = np.full(
            (self.coordinates.shape[0],),
            self.weights,
            dtype=np.float32,
        )

    # @profile
    def _make_gaussian_kernel(self):
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
                Can be sparse. Defaults to None.

        Returns:
            tp.Tuple[np.ndarray, np.ndarray]:
                Values and indices of the kernel (suitable for a sparse tensor).
        """

        center = self.center - self.crop
        size = self.size.astype(np.int32)

        # Get the rows and columns for the kernel
        (rows, cols, idx, xs, ys) = self.make_shape(
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

    def _make_gabor_kernel(self):
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
