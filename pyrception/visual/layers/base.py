import typing as tp

# --------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
from pyrception.utils import functions as pf
from pyrception.utils.logging import Logger
from pyrception.visual import ReceptiveFields
from pyrception.visual.utils.types import ImagePlot
from pyrception.visual.utils.types import KernelFilter


class BaseLayer(Logger):
    """
    Simple layer implementing some basic methods used by all retinal layers.
    """

    def __init__(
        self,
        shape: tp.Tuple[int, ...],
        name: str = "Base layer",
        notifier: tp.Callable = None,
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
        container: tp.Any,
    ) -> np.ndarray:
        """
        Flatten a potentially nested (heterogeneous) array of elements.

        Args:
            container (tp.Any):
                A container or a numeric value.

        Returns:
            np.ndarray:
                A NumPy array.
        """
        array = []
        if isinstance(container, (int, float)):
            array.append(container)

        elif isinstance(container, tp.Iterable):
            for element in container:
                if isinstance(element, np.ndarray):
                    element = element.tolist()
                array.extend(self._flatten(element))

        return np.unique(np.array(array, dtype=np.int32))

    def _to_rgba(
        self,
        colour: tp.Union[str, tp.Iterable],
    ) -> np.ndarray:
        """
        Convert a colour specified as HEX into RGBA (a 4-tuple).

        Args:
            colour (tp.Union[str, tp.Iterable]):
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
        cells: tp.Union[tp.List[int], tp.Tuple[int]] = None,
        cell_colour: tp.Tuple[str, tp.Tuple[int, ...]] = "#00ffff",
        rf_colour: tp.Tuple[str, tp.Tuple[int, ...]] = "#ff00ff",
        title: str = "Receptive fields",
        figsize: tp.Tuple = (8, 6),
        canvas: np.ndarray = None,
        fig: plt.Figure = None,
        axes: plt.Axes = None,
        spines: bool = False,
    ) -> tp.Tuple[plt.Figure, plt.Axes, tp.List, np.ndarray]:
        """
        Plot the receptive fields of amacrine cells.
        This takes into account the sparsity of bipolar cells.

        Args:

            rfs: (ReceptiveFields):
                Receptive fields to plot.

            cells (tp.Union[tp.List[int], tp.Tuple[int]], optional):
                Coordinates of the cells to plot. Defaults to None.

            cell_colour (tp.Tuple[str, tp.Tuple[int, ...]], optional):
                The colour to use for highlighting the plotted cells.
                Defaults to

            rf_colour (tp.Tuple[str, tp.Tuple[int, ...]], optional):
                The colour to use for highlighting the plotted receptive field.

            title (str, optional):
                Plot title. Defaults to "Amacrine cell receptive fields".

            figsize (tp.Tuple, optional):
                Figure size. Defaults to (8, 6).

            fig (plt.Figure, optional):
                tp.Optional preexisting Figure instance. Defaults to None.

            axes (plt.Axes, optional):
                tp.Optional preexisting Axes instance. Defaults to None.

        Returns:
            tp.Tuple[plt.Figure, plt.Axes, tp.List]:
                A tuple containing:
                    1. A Figure object.
                    2. An Axes object.
                    3. A list of mappables (which can be used for animations).
        """
        # Bipolar cell input for a single amacrine cell
        if canvas is None:
            canvas = np.zeros((rfs.height, rfs.width, 3))

        elif len(canvas.shape) == 2:
            canvas = np.vstack([canvas] * 3)

        # Convert HEX colours to RGB
        # ==================================================
        if cell_colour is not None:
            cell_colour = self._to_rgba(cell_colour)[:-1]
        rf_colour = self._to_rgba(rf_colour)[:-1][None, :]

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

                if rfs.kernel_params.filter == KernelFilter.Gaussian:
                    v = rfs.rf_vals[c]
                    v = (v - v.min()) / (v.max() - v.min())

                else:
                    v = np.ones_like(rfs.rf_vals[c])

                canvas[
                    rfs.rf_rows[c],
                    rfs.rf_cols[c],
                ] += rf_colour


        # Normalise

        # Plot the cells last so that they are superimposed
        # on top of the receptive fields.
        # ==================================================
        if cell_colour is not None:
            for c in cells:
                canvas[
                    rfs.cell_coordinates[c, 0],
                    rfs.cell_coordinates[c, 1],
                ] = cell_colour * canvas.max()

        canvas /= canvas.max()

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
        rfs: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        """
        Convolve the unrolled input vector with the current layer's receptive field.

        Args:
            frame (np.ndarray):
                The input frame, unrolled into a 1D vector.

        Returns:
            np.ndarray:
                The convolved frame.
        """

        # TODO
        # Explore alternatives for matrix operations (CuPy?)
        # ==================================================
        return rfs @ vector
