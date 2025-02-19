import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
from pyrception.utils.logging import LoggingMixin
from pyrception.utils.types import KernelFilter
from pyrception.utils.types import KernelShape


class Kernel(LoggingMixin):

    def __init__(
        self,
        size: np.ndarray,
        center: np.ndarray,
        fov: np.ndarray = None,
        crop: np.ndarray = None,
        shape: KernelShape = KernelShape.Elliptic,
        filter: KernelFilter = KernelFilter.Uniform,
        angle: float = 0.0,
        index_map: np.ndarray = None,
        substrate: np.ndarray = None,
        params: dict = None,
        weights: np.ndarray = None,
    ):

        super().__init__("Kernel")
        if params is None:
            params = {}

        if crop is None:
            crop = np.array([0, 0], dtype=np.uint32)
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

        self.coordinates = None
        self.indices = None
        self.outline = None
        self.weights = weights

        self.make_shape()
        self.extract_indices(index_map, substrate)
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
            KernelFilter.Uniform: self._make_uniform_weights,
            KernelFilter.Gaussian: self._make_gaussian_weights,
            KernelFilter.Gabor: self._make_gabor_weights,
        }

        filter_function = filter_functions.get(self.filter, None)

        if filter_function is None:
            raise TypeError(f"Invalid kernel filter '{self.filter}'")

        return filter_function

    def extract_indices(
        self,
        index_map: np.ndarray,
        substrate: np.ndarray,
    ):
        """
        Extract the sparse indices of the kernel based on the
        arrangement of input cells in the substrate.

        Args:
            index_map (np.ndarray):
                A 2D array containing the indices of each cell in the substrate.

            substrate (np.ndarray):
                The input substrate.
        """

        # Indices
        # ==================================================
        overlap = index_map[self.coordinates[:, 0], self.coordinates[:, 1]]
        self.indices = overlap[overlap != 0]
        # print(f"==[ indices: {self.indices}")

        # Re-extract the coordinates from the indices and the substrate.
        # This is necessary to sparsify the coordinates for layers with sparse input.
        # print(f"==[ centre: {self.center}")
        # print(f"==[ len(substrate): {len(substrate)}")
        self.coordinates = substrate[self.indices]

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
        """

        center = self.center - self.crop

        # Coordinates
        # ==================================================
        self.coordinates = (
            np.stack(
                ski.draw.rectangle(
                    center - self.size,
                    center + self.size,
                    shape=self.fov,
                ),
                axis=2,
            )
            + self.crop
        ).reshape(-1,2)

        # Outline
        # ==================================================
        self.outline = (
            np.stack(
                ski.draw.rectangle_perimeter(
                    center - self.size,
                    center + self.size,
                    shape=self.fov,
                ),
                axis=1,
            )
            + self.crop
        )

    def _make_uniform_weights(self):
        """
        Weights from a uniform distribution.
        """

        if self.coordinates.size == 0:
            return

        if self.weights is None:
            self.weights = np.full(
                (len(self.coordinates),),
                1 / len(self.coordinates),
                dtype=np.float32,
            )

    def _make_gaussian_weights(self):
        """
        Weights following a 2D Gaussian distribution.
        """

        center = self.center - self.crop
        size = self.size.astype(np.int32)

        sd = size * 0.5

        # Create a Gaussian distribution
        coords = (self.coordinates - center) / (sd)

        self.weights = np.exp(-0.5 * (coords[:, 0] ** 2 + coords[:, 1] ** 2))

        # Normalise if necessary
        if self.params.get("normalise"):
            self.weights /= 2 * np.pi * (sd[0] * sd[1])

    def _make_gabor_weights(self):
        """
        Weights corresponding to a Gabor filter.

        WIP: This is a stub.
        TODO: Check if we can incorporate this into the Gaussian kernel method
        since we only have to add a couple of extra parameters and superimpose
        the sine funciton.
        """

        pass
