from typing import *

# --------------------------------------
from datetime import datetime

# --------------------------------------
from pathlib import Path

# --------------------------------------
import shutil

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt

# --------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update(
    {
        "figure.figsize": (10, 10),
        "figure.frameon": False,
        "figure.facecolor": "black",
    }
)

# --------------------------------------
import pyrception.util.functions as pcf
from pyrception.visual.util.types import PlotEntry
from pyrception.visual.util.types import ImagePlot
from pyrception.visual.util.types import ScatterPlot


def cwd(path: Union[Path, str]):
    path = Path(path).resolve().absolute()
    if not path.exists():
        return None
    return path.parent if path.is_file() else path


def timestamp(ms: bool = False) -> str:
    """
    Create a timestamp string.
    """

    fmt = "%Y-%m-%d_%H-%M-%S"
    end = None

    if ms:
        # With ms precision
        fmt += ":%f"
        end = -3

    return datetime.strftime(datetime.utcnow(), fmt)[:end]


def plot(
    entries: List[List[PlotEntry]] = None,
    height: int = 8,
    width: int = 10,
    scale: int = 1,
    figsize: Tuple[int] = None,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    animated: bool = False,
    cmap: str = None,
    dpi: int = 96,
) -> Tuple[plt.Figure, plt.Axes, List]:
    """
    Plot images in a row, column or grid pattern.

    Args:
        entries (List[List[Dict[str, Any]]], optional):
            Data items to plot. Defaults to None.

        height (int, optional):
            Height of a single plot. Defaults to 8.

        width (int, optional):
            Width of a single plot. Defaults to 10.

        scale (int, optional):
            Scale of the plot. Defaults to 1.

        figsize (Tuple[int], optional):
            Figure size. Defaults to None.
            If this is not provided, the figure size is computed from the width and height.

        fig (plt.Figure, optional):
            Optional preexisting Figure instance. Defaults to None.

        axes (plt.Axes, optional):
            Optional preexisting Axes instance. Defaults to None.

        animated (bool, optional):
            Toggle indicating if the plot would be used for animation. Defaults to False.

        cmap (str, optional):
            The colour map to use for plots. Defaults to None.

        dpi (int, optional):
            DPI setting. Defaults to 96.
            Used for computing the figure size from the width and the height.

    Returns:
        Tuple[plt.Figure, plt.Axes, List]:
            A tuple containing:
                1. A Figure object.
                2. An Axes object.
                3. A list of mappables (which can be used for animations).
    """
    rows = 1
    cols = 1

    if entries is None:
        entries = []
    else:
        if isinstance(entries, (PlotEntry, np.ndarray, pt.Tensor)):
            entries = [[entries]]

        elif isinstance(entries[0], (PlotEntry, np.ndarray, pt.Tensor)):
            entries = [entries]

        rows = len(entries)
        cols = max([len(row) for row in entries])

    if figsize is None:
        figsize = (scale * cols * height / dpi, scale * rows * width / dpi)

    fp = FontProperties()
    fs = fp.get_size()
    mappables = []

    with mpl.rc_context({"font.size": np.log(1 + scale) * fs}):

        if axes is None:
            fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

        for ridx, row in enumerate(entries):
            for cidx, entry in enumerate(row):

                if isinstance(entry, (np.ndarray, pt.Tensor)):
                    # Expand shortcut entries.
                    # Assume a scatter plot if the array is 1D,
                    # otherwise assume an image.
                    entry = ScatterPlot(entry) if len(entry.shape) == 1 else ImagePlot(entry)

                if entry.plottype is None:
                    raise ValueError(f"Invalid plot type for entry {entry}.")

                if rows == 1:
                    ax = axes if cols == 1 else axes[cidx]
                else:
                    ax = axes[ridx] if cols == 1 else axes[ridx, cidx]

                if isinstance(cmap, str):
                    cmap = cm[cmap]

                if entry.axis:
                    ax.axis("on")
                else:
                    ax.xaxis.set_ticks([])
                    ax.yaxis.set_ticks([])

                ax.spines[:].set_visible(entry.spines)

                if isinstance(entry, ImagePlot):
                    mappable = ax.imshow(
                        entry.data,
                        cmap=cmap,
                        animated=animated,
                        vmin=entry.clim[0],
                        vmax=entry.clim[1],
                    )

                elif isinstance(entry, ScatterPlot):
                    (mappable,) = ax.plot(
                        pt.arange(len(entry.data)),
                        entry.data,
                        marker=entry.marker,
                        markersize=entry.size,
                        c=entry.colour,
                        linestyle="None",
                        animated=animated,
                    )

                if entry.colourbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(mappable, ax=ax, cax=cax)

                ax.set_xlabel(entry.xlabel)
                ax.set_ylabel(entry.ylabel)
                ax.set_title(entry.title)
                mappables.append(mappable)

        if fig is not None:
            fig.tight_layout()

        return (fig, axes, mappables)


def animate(
    fig: plt.Figure,
    animator: Callable,
    producer: Callable,
    interval: int = 1,
    format: str = "avi",
    title: str = "",
    fps: int = 30,
    output_dir: Path = Path("./"),
) -> Tuple[FuncAnimation, Path]:
    """
    Create an animation from a figure that is being continually updated.

    Args:
        fig (plt.Figure):
            The figure being animated.

        animator (Callable):
            A function that produces the animation by updating the figure.

        producer (Callable):
            A functon that generates frames.

        interval (int, optional):
            The delay between each frame. Defaults to 1.

        format (str, optional):
            Animation file format (used for the writer). Defaults to "avi".

        title (str, optional):
            Title to be displayed on each frame (also serves as a file name). Defaults to "".

        fps (int, optional):
            Frames per second. Defaults to 30.

        output_dir (Path, optional):
            Directory where the animation is saved. Defaults to Path("./") (the current directory).

    Returns:
        Tuple[FuncAnimation, Path]:
            A tuple containing:
                1. The animation object.
                2. The path to the saved animation.
    """

    ts = pcf.timestamp()
    filename = Path(f"{title.lower().replace(' ','_')}-{ts}")

    try:

        ani = FuncAnimation(
            fig,
            animator,
            producer,
            interval=interval,
            blit=True,
            cache_frame_data=False,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (output_dir / filename).resolve().absolute()
        filename = filename.with_suffix(f".{format}")

        if format == "gif":
            writer = PillowWriter(fps=fps, metadata=dict(artist=title))
        elif format == "avi":
            writer = FFMpegWriter(fps=fps, codec="ffv1")

        ani.save(filename, writer=writer)
    except Exception as e:

        print(f"==[ Error: {e}")
        ani = None
        filename.unlink(missing_ok=True)
        filename = None

    return ani, filename
