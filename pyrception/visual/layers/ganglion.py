from typing import *

# --------------------------------------
import torch as pt

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from pyrception.util import functions as pf
from pyrception.visual.layers.bipolar import BipolarLayer
from pyrception.visual.layers.amacrine import AmacrineLayer
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.rf import ReceptiveFields


class GanglionLayer(BaseLayer):
    """
    A layer of ganglion cells receiving excitatory input from
    bipolar cells and inhibitory input from amacrine cells.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        bipolar: BipolarLayer,
        amacrine: AmacrineLayer,
        sectors: int = 32,
        name: str = "Ganglion",
        bipolar_params: Dict[str, Any] = None,
        amacrine_params: Dict[str, Any] = None,
    ):

        # Initialise the base
        super().__init__(size, name)

        # Initialise the bipolar receptive fields.
        # ==================================================
        self.bipolar = bipolar
        if bipolar_params is None:
            bipolar_params = {}
        bipolar_params.setdefault("name", f"{name} | Bipolar RFs")
        bipolar_params["sectors"] = sectors
        self.bipolar_rfs = ReceptiveFields(
            self.size,
            bipolar.rfs.cell_coordinates,
            **bipolar_params,
        )
        self.bipolar_rfs.make_rfs()

        # Initialise the amacrine receptive fields.
        # ==================================================
        self.amacrine = amacrine
        if amacrine_params is None:
            amacrine_params = {}
        amacrine_params.setdefault("name", f"{name} | Amacrine RFs")
        amacrine_params["sectors"] = sectors
        self.amacrine_rfs = ReceptiveFields(
            self.size,
            amacrine.rfs.cell_coordinates,
            **amacrine_params,
        )
        self.amacrine_rfs.make_rfs()

        # ON/OFF and OFF/ON ganglion cells
        # self.on_off =

        self.info("Initialised.")

    def forward(self):
        """
        Compute the activation of ON/OFF and OFF/ON RGCs.

        This is where spikes are produced.
        """

        # ON centre / OFF surround
        on_off = self.convolve(
            self.bipolar_rfs.rfs,
            self.bipolar.on,
        ) - self.convolve(
            self.amacrine_rfs.rfs,
            self.amacrine.off,
        )
        on_off_spikes = pt.where(on_off >= 0, 1, 0)

        print(f"==[ on_off: {on_off} ({on_off_spikes.sum().int()}/{len(on_off_spikes)})")

        # OFF centre / ON surround
        off_on = self.convolve(
            self.bipolar_rfs.rfs,
            self.bipolar.off,
        ) - self.convolve(
            self.amacrine_rfs.rfs,
            self.amacrine.on,
        )
        off_on_spikes = pt.where(off_on >= 0, 1, 0)
        print(
            f"==[ on_off: {off_on} ({off_on_spikes.sum().int()}/{len(off_on_spikes)})"
        )

        return (on_off_spikes, off_on_spikes)

    def plot_rfs(
        self,
        bipolar_rf_colour: str = "#ff00ffff",
        amacrine_rf_colour: int = "#00ff00ff",
        *args,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes, List, np.ndarray]:
        """
        Plot the receptive fields of amacrine cells.
        This takes into account the sparsity of bipolar cells.

        Args:

            bipolar_rf_colour (int, optional):
                The value to use for highlighting the plotted bipolar cells.

            amacrine_rf_colour (int, optional):
                The value to use for highlighting the plotted amacrine cells.

        Returns:
            Tuple[plt.Figure, plt.Axes, List]:
                A tuple containing:
                    1. A Figure object.
                    2. An Axes object.
                    3. A list of mappables (which can be used for animations).
                    3. The canvas.
        """

        kwargs.setdefault("title", "Ganglion layer receptive fields")

        # Plot the bipolar cells
        (fig, axes, _, canvas) = self._plot_rfs(
            self.bipolar_rfs,
            rf_colour=bipolar_rf_colour,
            *args,
            **kwargs,
        )

        # Plot the amacrine cells
        (fig, axes, _, _) = self._plot_rfs(
            self.amacrine_rfs,
            rf_colour=amacrine_rf_colour,
            canvas=canvas,
            fig=fig,
            axes=axes,
            *args,
            **kwargs,
        )

        return (fig, axes, _, canvas)
