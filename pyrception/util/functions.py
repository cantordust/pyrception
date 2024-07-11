import typing as tp

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


def cwd(path: tp.Union[Path, str]):
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
    entries: tp.List[tp.List[PlotEntry]] = None,
    figsize: tp.Tuple[int, int] = (12, 8),
    height: int = None,
    width: int = None,
    title: str = None,
    scale: int = 1,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    animated: bool = False,
    dpi: int = 96,
) -> tp.Tuple[plt.Figure, plt.Axes, tp.List]:
    """
    Plot images in a row, column or grid pattern.

    Args:
        entries (tp.List[tp.List[tp.Dict[str, tp.Any]]], optional):
            Data items to plot. Defaults to None.

        figsize (tp.Tuple[int], optional):
            Figure size. Defaults to None.
            If this is not provided, the figure size is computed from the width and height.

        height (int, optional):
            Height of a single plot. Defaults to 8.

        width (int, optional):
            Width of a single plot. Defaults to 10.

        title (str, optional):
            Figure title. Defaults to None.

        scale (int, optional):
            Scale of the plot. Defaults to 1.

        fig (plt.Figure, optional):
            tp.Optional preexisting Figure instance. Defaults to None.

        axes (plt.Axes, optional):
            tp.Optional preexisting Axes instance. Defaults to None.

        animated (bool, optional):
            Toggle indicating if the plot would be used for animation. Defaults to False.

        dpi (int, optional):
            DPI setting. Defaults to 96.
            Used for computing the figure size from the width and the height.

    Returns:
        tp.Tuple[plt.Figure, plt.Axes, tp.List]:
            A tuple containing:
                1. A Figure object.
                2. An Axes object.
                3. A list of mappables (which can be used for animations).
    """
    rows = 1
    cols = 1

    if title is None:
        title = ""

    if entries is None or (isinstance(entries, tp.List) and len(entries) == 0):
        raise ValueError(f"Please provide at least one entry to plot")
    else:
        if isinstance(entries, (ImagePlot, ScatterPlot, np.ndarray, pt.Tensor)):
            entries = [[entries]]

        elif isinstance(entries[0], (ImagePlot, ScatterPlot, np.ndarray, pt.Tensor)):
            entries = [entries]

        rows = len(entries)
        cols = max([len(row) for row in entries])

    if figsize is None:
        if height is None or width is None:
            raise ValueError(
                f"Please specify either the figsize (in inches) or both the height and the width (in pixels)"
            )
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
                    entry = (
                        ScatterPlot(entry)
                        if len(entry.shape) == 1
                        else ImagePlot(entry)
                    )

                if entry.plottype is None:
                    raise ValueError(f"Invalid plot type for entry {entry}.")

                if rows == 1:
                    ax = axes if cols == 1 else axes[cidx]
                else:
                    ax = axes[ridx] if cols == 1 else axes[ridx, cidx]

                if hasattr(entry, "cmap") and isinstance(entry.cmap, str):
                    entry.cmap = cm[entry.cmap]

                if entry.axis:
                    ax.axis("on")
                    entry.spines = True
                else:
                    ax.xaxis.set_ticks([])
                    ax.yaxis.set_ticks([])

                ax.spines[:].set_visible(entry.spines)

                if isinstance(entry, ImagePlot):
                    kwargs = {}
                    if entry.norm is None:
                        kwargs.update(
                            vmin=entry.data.min(),
                            vmax=entry.data.max(),
                        )
                    else:
                        kwargs.update(norm=entry.norm)

                    mappable = ax.imshow(
                        entry.data,
                        cmap=entry.cmap,
                        animated=animated,
                        **kwargs,
                    )

                    if entry.colourbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        fig.colorbar(mappable, ax=ax, cax=cax)

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

                ax.set_xlabel(entry.xlabel)
                ax.set_ylabel(entry.ylabel)
                ax.set_title(entry.title)
                mappables.append(mappable)

        fig.suptitle(title)

        if fig is not None:
            fig.tight_layout()

        return (fig, axes, mappables)


def animate(
    fig: plt.Figure,
    animator: tp.Callable,
    producer: tp.Callable,
    interval: int = 1,
    format: str = "avi",
    title: str = "",
    fps: int = 30,
    output_dir: Path = Path("./"),
) -> tp.Tuple[FuncAnimation, Path]:
    """
    Create an animation from a figure that is being continually updated.

    Args:
        fig (plt.Figure):
            The figure being animated.

        animator (tp.Callable):
            A function that produces the animation by updating the figure.

        producer (tp.Callable):
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
        tp.Tuple[FuncAnimation, Path]:
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
            blit=False,
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
        elif format == "mp4":
            writer = FFMpegWriter(fps=fps, codec="libx264")

        ani.save(filename, writer=writer)
    except Exception as e:

        print(f"==[ Error: {e}")
        ani = None
        filename.unlink(missing_ok=True)
        filename = None

    return ani, filename
