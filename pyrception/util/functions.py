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


cmap = cm["gray"]


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
    entries: List[List[Dict[str, Any]]] = None,
    height: int = 8,
    width: int = 10,
    scale: int = 1,
    figsize: Tuple[int] = None,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    spines: bool = False,
    animated: bool = False,
    dpi: int = 96,
):

    rows = 1
    cols = 1

    if entries is None:
        entries = []
    if len(entries) > 0:
        if isinstance(entries, (Dict, np.ndarray, pt.Tensor)):
            entries = [[entries]]

        elif isinstance(entries[0], (Dict, np.ndarray, pt.Tensor)):
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
                    # Expand shortcut entries
                    entry = {"data": entry}

                data = entry["data"]
                plottype = entry.get("type", "image")
                entry.setdefault("axis", False)
                entry.setdefault("colorbar", False)
                entry.setdefault("clim", (None, None))

                if plottype == "scatter":
                    entry.setdefault("c", "#00ffff")
                    entry.setdefault("s", 0.1)

                if rows == 1:
                    ax = axes if cols == 1 else axes[cidx]
                else:
                    ax = axes[ridx] if cols == 1 else axes[ridx, cidx]

                ax.spines["top"].set_visible(spines)
                ax.spines["right"].set_visible(spines)
                ax.spines["bottom"].set_visible(spines)
                ax.spines["left"].set_visible(spines)

                if plottype == "image":
                    mappable = ax.imshow(
                        data,
                        cmap=cmap,
                        animated=animated,
                        vmin=entry['clim'][0],
                        vmax=entry['clim'][1],
                    )

                elif plottype == "scatter":
                    mappable = ax.scatter(
                        pt.arange(len(data)),
                        data,
                        s=entry["s"],
                        c=entry["c"],
                        animated=animated,
                    )

                if not entry["axis"]:
                    ax.axis("off")

                if entry["colorbar"]:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(mappable, ax=ax, cax=cax)

                if "xlabel" in entry:
                    ax.set_xlabel(entry["xlabel"])

                if "ylabel" in entry:
                    ax.set_ylabel(entry["ylabel"])

                if "title" in entry:
                    ax.set_title(entry["title"])

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
):
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
