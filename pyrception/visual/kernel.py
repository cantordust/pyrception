from typing import Any

import numpy as np
import skimage as ski

from pyrception import utils
from pyrception.utils.enums import KernelShape
from pyrception.utils.enums import KernelFilter


class Kernel:

    def __init__(
        self,
        size: int | tuple[int, ...] | np.ndarray,
        center: tuple[int, ...] | np.ndarray,
        shape: KernelShape = KernelShape.Elliptic,
        filter: KernelFilter = KernelFilter.Uniform,
        angle: float = 0.0,
        index: int | None = None,
        coords: np.ndarray | None = None,
        outline: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        span: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        name: str = "Kernel",
        **params,
    ):

        if len(size) > 2 or any(s <= 0 for s in size):
            raise AttributeError(f"Invalid size '{size}'")

        self.size = utils.arg2np(size, dtype=np.int32, ext=2)
        self.center = utils.arg2np(center, dtype=np.int32, ext=2)
        self.shape = KernelShape(shape)
        self.filter = KernelFilter(filter)
        self.angle = float(angle)
        self.index = index
        self.coords = coords
        self.outline = outline
        self.weights = weights
        self.span = span
        self.offset = offset
        self.name = name
        self.params = params or {}

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(size={self.size}, shape={self.shape}, filter={self.filter})"

    @staticmethod
    def make_kernel(
        kshape: KernelShape,
        size: np.ndarray,
        angle: float = 0.0,
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        """
        Create a kernel with the requested shape and angle.

        Args:
            kshape:
                A KernelShape parameter.

            size:
                Kernel size (x and y extent).

            angle:
                Rotation angle.

        Returns:
            The kernel and its outline, each as two separate arrays
            (rows and columns).

        Raises:
            TypeError:
                Raised if the requested kernel shape is invalid.
        """
        match kshape:
            case KernelShape.Rectangular:
                return Kernel.make_rectangular_kernel(size, angle)

            case KernelShape.Elliptic:
                return Kernel.make_elliptic_kernel(size, angle)

            case _:
                raise TypeError(f"Invalid kernel shape '{kshape}'")

    @staticmethod
    def make_filter(
        kfilter: KernelFilter,
        rows: np.ndarray,
        cols: np.ndarray,
        **params,
    ) -> np.ndarray:
        """
        Compute the weights that implement the specified filter
        from the row and column indices of the kernel.

        Args:
            kfilter:
                Filter type (uniform, Gaussian, etc.)

            rows:
                Row indices to the kernel.

            cols:
                Column indices of the kernel.

        Raises:
            TypeError:
                Raised if the requested kernel filter is invalid.

        Returns:
            The kernel factory function for the requested kernel filter.
        """

        match kfilter:
            case KernelFilter.Uniform:
                return Kernel.make_uniform_filter(rows, **params)

            case KernelFilter.Gaussian:
                return Kernel.make_gaussian_filter(rows, cols, **params)

            case KernelFilter.Gabor:
                return Kernel.make_gabor_filter(rows, cols, **params)

            case _:
                raise TypeError(f"Invalid filter type '{kfilter}'")

    @staticmethod
    def _rotate(
        rows: np.ndarray,
        cols: np.ndarray,
        angle: float,
    ):
        """
        Rotate the indices (rows and columns) by a certain angle.

        Args:
            rows:
                Row part of the indices.

            cols:
                Column part of the indices.

            angle:
                Angle to rotate by.

        Returns:
            The rotated indices (rounded to an integer so that they can be used
            to index other arrays).
        """

        angle = np.deg2rad(-angle)
        c, s = np.cos(angle), np.sin(angle)
        _rows = rows * c + cols * s
        _cols = -rows * s + cols * c

        return np.round(_rows).astype(np.int32), np.round(_cols).astype(np.int32)

    @staticmethod
    def make_elliptic_kernel(
        size: np.ndarray,
        angle: float = 0.0,
    ) -> np.ndarray:
        """
        Create an elliptic kernel having a certain size
        (defined as the semi-major axes).
        The kernel can be optionally rotated.

        NOTE: We assume that rotation angles increase *counterclockwise*.

        Args:
            size:
                Kernel size (sides of the rectangle).

            angle:
                Rotation angle.

        Returns:
            The rows and columns of the kernel and its outline.
        """

        angle = np.deg2rad(angle)

        size = np.clip(size, 0.5, None)

        coords = ski.draw.ellipse(
            0,
            0,
            size[0],
            size[1],
            rotation=angle,
        )

        outline = ski.draw.ellipse_perimeter(
            0,
            0,
            size[0].astype(np.int32),
            size[1].astype(np.int32),
            orientation=-angle,
        )

        return coords, outline

    @staticmethod
    def make_rectangular_kernel(
        size: np.ndarray,
        angle: float = 0.0,
    ) -> np.ndarray:
        """
        Create a rectangular kernel and its outline.

        NOTE: We assume that rotation angles increase *counterclockwise*.

        Args:
            size:
                Kernel size (sides of the rectangle).

            angle:
                Rotation angle.

        Returns:
            The rows and columns of the kernel and its outline.
        """

        # Compute the extent of the rectangle in the x and y directions
        size = np.repeat(size[None, :], 2, axis=0).astype(np.int32)
        size[0] *= -1
        xext, yext = size[:, 0], size[:, 1]

        # Vertex indices
        vr = np.repeat(xext, 2)
        vc = np.concatenate((yext, yext[::-1]))

        # Rotate
        vr, vc = Kernel._rotate(vr, vc, angle)

        # Dimensions of the virtual container of the polygon.
        # Any negative indices are dropped by SKImange by default,
        # so we need to shift by the minimum in each dimension.
        h = vr.max() - vr.min()
        w = vc.max() - vc.min()
        vr += abs(vr.min())
        vc += abs(vc.min())

        # Draw the rectangles
        rr, cc = ski.draw.polygon(vr, vc)
        rrp, ccp = ski.draw.polygon_perimeter(vr, vc)

        # Shift back by h // 2 and w // 2 to center at the origin.
        coords = (rr - h // 2, cc - w // 2)
        outline = (rrp - h // 2, ccp - w // 2)

        return coords, outline

    @staticmethod
    def make_uniform_filter(
        rows: np.ndarray,
        **params,
    ) -> np.ndarray:
        """
        Weights following a uniform distribution.

        Args:
            rows:
                Kernel rows.

        Returns:
            The kernel weights.
        """

        return np.full((len(rows),), 1 / len(rows), dtype=np.float32)

    @staticmethod
    def make_gaussian_filter(
        rows: np.ndarray,
        cols: np.ndarray,
        angle: float = 0.0,
        sd: float = (1.0, 1.0),
        **params,
    ) -> np.ndarray:
        """
        Weights following a 2D Gaussian distribution.

        Args:
            rows:
                Kernel rows.

            cols:
                Kernel columns.

            angle:
                Kernel rotation angle.

            sd:
                The standard deviation (the x and y values can be specified independently).

        Returns:
            The normalised weights of the kernel.
        """

        if isinstance(sd, (float, int)) or len(sd) == 1:
            sd = (sd, sd)

        # Rotate
        _rows, _cols = Kernel._rotate(rows, cols, -angle)

        sd_x = sd[0] * np.clip(_cols.std(), 0.1, None)
        sd_y = sd[1] * np.clip(_rows.std(), 0.1, None)

        # Make a Gaussian
        weights = np.exp(-0.5 * ((_rows / sd_y) ** 2 + (_cols / sd_x) ** 2))

        # Normalise
        weights /= 2 * np.pi * (sd_y * sd_x)

        return weights / weights.sum()

    @staticmethod
    def make_gabor_filter(
        rows: np.ndarray,
        cols: np.ndarray,
        angle: float = 0.0,
        sd: float = (1.0, 1.0),
    ):
        """
        Weights corresponding to a Gabor filter.

        TODO: Check if we can incorporate this into the Gaussian kernel method
        since we only have to add a couple of extra parameters and superimpose
        the sine funciton.
        """
        pass

    @staticmethod
    def make_von_mises_filter(
        rows: np.ndarray,
        cols: np.ndarray,
        angle: float = 0.0,
        sd: float = (1.0, 1.0),
    ):
        """
        Weights corresponding to a Gabor filter.

        TODO: Check if we can incorporate this into the Gaussian kernel method
        since we only have to add a couple of extra parameters and superimpose
        the sine funciton.
        """
        pass
