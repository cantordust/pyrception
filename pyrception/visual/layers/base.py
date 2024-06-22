from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --------------------------------------
import numpy as np

# --------------------------------------
from pyrception.logging import Logging
from pyrception.util import functions as pf
from pyrception.visual import ReceptiveFields
from pyrception.visual.util.types import ImagePlot


class BaseLayer(Logging):
    """
    Simple layer implementing some basic methods used by all retinal layers.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        name: str = "Base layer",
    ):
        super().__init__(name)

        self.info("Initialising...")

        # Check if the layer size is valid.
        # Order: height, width, depth
        if len(size) not in (2, 3):
            raise ValueError(
                f"The layer size must have 2 or 3 dimensions ({len(size)} provided)."
            )
        if len(size) == 2:
            # Add a depth of 1 if it is missing
            size += (1,)

        self.size = size
        self.name = name

    def _flatten(
        self,
        container: Any,
    ) -> np.ndarray:
        """
        Flatten a potentially nested (heterogeneous) array of elements.

        Args:
            container (Any):
                A container or a numeric value.

        Returns:
            np.ndarray:
                A NumPy array.
        """
        array = []
        if isinstance(container, (int, float)):
            array.append(container)

        elif isinstance(container, Iterable):
            for element in container:
                if isinstance(element, pt.Tensor):
                    element = element.tolist()
                array.extend(self._flatten(element))

        return np.unique(np.array(array, dtype=np.int32))

    def _to_rgba(
        self,
        colour: Union[str, Iterable],
    ) -> np.ndarray:
        """
        Convert a colour specified as

        Args:
            colour (Union[str, Iterable]):
                The colour specified as a HEX string or an iterable.

        Returns:
            np.ndarray:
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
        rfs: ReceptiveFields,
        cells: Union[List[int], Tuple[int]] = None,
        cell_colour: Tuple[str, Tuple[int, ...]] = "#00ffffff",
        rf_colour: Tuple[str, Tuple[int, ...]] = "#ff00ff33",
        title: str = "Receptive fields",
        figsize: Tuple = (8, 6),
        canvas: np.ndarray = None,
        fig: plt.Figure = None,
        axes: plt.Axes = None,
        spines: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes, List, np.ndarray]:
        """
        Plot the receptive fields of amacrine cells.
        This takes into account the sparsity of bipolar cells.

        Args:

            rfs: (ReceptiveFields):
                Receptive fields to plot.

            cells (Union[List[int], Tuple[int]], optional):
                Coordinates of the cells to plot. Defaults to None.

            cell_colour (Tuple[str, Tuple[int, ...]], optional):
                The colour to use for highlighting the plotted cells.
                Defaults to

            rf_colour (Tuple[str, Tuple[int, ...]], optional):
                The colour to use for highlighting the plotted receptive field.

            title (str, optional):
                Plot title. Defaults to "Amacrine cell receptive fields".

            figsize (Tuple, optional):
                Figure size. Defaults to (8, 6).

            fig (plt.Figure, optional):
                Optional preexisting Figure instance. Defaults to None.

            axes (plt.Axes, optional):
                Optional preexisting Axes instance. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes, List]:
                A tuple containing:
                    1. A Figure object.
                    2. An Axes object.
                    3. A list of mappables (which can be used for animations).
        """
        # Bipolar cell input for a single amacrine cell
        if canvas is None:
            canvas = np.zeros((rfs.height, rfs.width, 4))

        elif len(canvas.shape) == 2:
            canvas = np.vstack([canvas] * 4)

        elif len(canvas) == 3:
            canvas = np.vstack((canvas, np.ones((canvas.shape[0], canvas.shape[1]))))

        # Convert HEX colours to RGBA
        # ==================================================
        cell_colour = self._to_rgba(cell_colour)
        rf_colour = self._to_rgba(rf_colour)

        # Process the requested cell coordinates.
        # ==================================================
        cells = (
            np.arange(len(rfs.cell_coordinates), dtype=np.int32)
            if cells is None
            else self._flatten(cells)
        )

        # Plot the receptive field
        # ==================================================
        if rf_colour is not None:
            for c in cells:
                canvas[rfs.rows[c], rfs.cols[c]] += rf_colour

                # # Scale by the minimal RF factor
                # if rfs.rf_factors is not None:
                #     canvas[rfs.cols[c], rfs.rows[c]] *= rfs.rf_factors.numpy().min()
                # else:
                #     canvas[rfs.cols[c], rfs.rows[c]] /= canvas.max()

            # Make the RF fully opaque
        # canvas[:, :, 3] = rf_colour[-1]

        # Plot the cells last so that they are superimposed
        # on top of the receptive fields.
        # ==================================================
        cell_coords = rfs.cell_coordinates
        if cell_colour is not None:
            for c in cells:
                canvas[
                    cell_coords[c, 0],
                    cell_coords[c, 1],
                ] = cell_colour

        # Now actually plot everything onto the canvas
        # ==================================================
        plot_params = ImagePlot(
            canvas,
            colourbar=False,
            title=title,
            spines=spines,
        )

        (fig, axes, mappables) = pf.plot(
            [plot_params],
            figsize=figsize,
            fig=fig,
            axes=axes,
        )

        return (fig, axes, mappables, canvas)

    def convolve(
        self,
        rfs: pt.Tensor,
        vector: pt.Tensor,
    ) -> pt.Tensor:
        """
        Convolve the unrolled input vector with the current layer's receptive field.

        Args:
            frame (pt.Tensor):
                The input frame, unrolled into a 1D vector.

        Returns:
            pt.Tensor:
                The convolved frame.
        """

        # TODO
        # Explore alternatives for matrix operations
        # (SciPy, CuPy, Trilinos...)
        # ==================================================
        return pt.mv(rfs, vector)
