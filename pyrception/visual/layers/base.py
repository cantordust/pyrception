from collections.abc import Callable
from collections.abc import Iterable

import numpy as np
import skimage as ski
from bokeh.plotting import figure
from matplotlib.colors import to_rgba

from typing import Any

from pyrception.utils.logging import LoggingMixin


class BaseLayer(LoggingMixin):
    """
    Simple layer implementing some basic methods used by all retinal layers.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        name: str = "Base layer",
        notifier: Callable = None,
    ):
        super().__init__(name, notifier)

        self.info("Initialising...")

        # Check if the layer shape is valid.
        # Order: height, width, depth
        if len(shape) not in (2, 3):
            raise ValueError(
                f"The layer size must have 2 or 3 dimensions ({len(shape)} provided)."
            )

        shape = tuple(shape)
        if len(shape) == 2:
            # Add a depth of 1 if it is missing
            shape += (1,)

        self.shape = shape
        self.name = name

    def _flatten(
        self,
        container: Iterable,
    ) -> np.ndarray:
        """
        Flatten a potentially nested (heterogeneous) array of elements.

        TODO: Move this into the `utils` module.

        Args:
            container:
                A container or a numeric value.

        Returns:
            A NumPy array.
        """
        array = []
        if isinstance(container, (int, float)):
            array.append(container)

        elif isinstance(container, Iterable):
            for element in container:
                if isinstance(element, np.ndarray):
                    element = element.tolist()
                array.extend(self._flatten(element))

        return np.unique(np.array(array, dtype=np.int32))

    def _to_rgba(
        self,
        colour: str | Iterable,
    ) -> np.ndarray:
        """
        Convert a colour specified as HEX into RGBA (a 4-tuple).

        TODO: Move this into the `utils` module.

        Args:
            colour:
                The colour specified as a HEX string or an iterable.

        Returns:
            The RGBA values of the colour.
        """
        if colour is not None:
            if isinstance(colour, str):
                colour = to_rgba(colour)
            colour = list(colour)
            if len(colour) == 3:
                colour.append(1.0)
            colour = np.array(colour)

        return colour

    def _plot_rfs(
        self,
        rfs: Any,
        cells: list[int] | tuple[int] = None,
        cell_colour: tuple[str, tuple[int, ...]] = "#00ffff",
        rf_colour: tuple[str, tuple[int, ...]] = "#ffffff",
        weighted: bool = False,
    ) -> figure:
        """
        Plot the receptive fields of amacrine cells.
        This takes into account the sparsity of bipolar cells.

        Args:

            rfs:
                Receptive fields to plot.

            cells:
                Coordinates of the cells to plot.

            cell_colour:
                The colour to use for highlighting the plotted cells.

            rf_colour:
                The colour to use for highlighting the plotted receptive field.

        Returns:
            A visualisation of the receptive fields.
        """
        # Bipolar cell input for a single amacrine cell
        canvas = np.zeros((rfs.height, rfs.width, 3))

        # Convert HEX colours to RGBA
        # ==================================================
        if cell_colour is not None:
            cell_colour = self._to_rgba(cell_colour)[:-1]
        rf_colour = self._to_rgba(rf_colour)[:-1][None, :]

        # Process the requested cell coordinates.
        # ==================================================
        cells = (
            np.arange(len(rfs.cell_coordinates), dtype=np.uint32)
            if cells is None
            else self._flatten(cells)
        )

        # Plot the receptive field
        # ==================================================
        if rf_colour is not None:
            for c in cells:
                colour = rf_colour
                if weighted:
                    # colour = colour * rfs.kernels[c].weights[:,None]
                    colour = colour * rfs.kernels[c].weights[:, None]

                canvas[
                    rfs.kernels[c].coordinates[:, 0],
                    rfs.kernels[c].coordinates[:, 1],
                ] += colour
                canvas = np.clip(canvas, max=5)

        canvas = ski.exposure.rescale_intensity(canvas, out_range=(0, 1))

        # Plot the cells last so that they are superimposed
        # on top of the receptive fields.
        # ==================================================
        if cell_colour is not None:
            for c in cells:
                canvas[
                    rfs.cell_coordinates[c, 0],
                    rfs.cell_coordinates[c, 1],
                ] = cell_colour

        return canvas

    def convolve(
        self,
        rfs: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        """
        Convolve the unrolled input vector with the current layer's receptive field.

        Args:
            frame:
                The input frame, unrolled into a 1D vector.

        Returns:
            The convolved frame.
        """

        # TODO
        # Explore alternatives for matrix operations (CuPy?)
        # ==================================================
        return rfs @ vector

    def update_state(self, dt: float, *args, **kwargs):
        """
        Updates the internal state based on temporal dynamics.
        Should be be implemented in derived classes.

        Args:
            dt:
                The time interval since the last input.
        """
        pass
