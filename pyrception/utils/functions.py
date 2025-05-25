from pathlib import Path
from datetime import datetime
import threading

import av
from tqdm import tqdm
import numpy as np
from IPython import get_ipython
import skimage as ski


def mkdir(path: Path | str) -> Path:
    """
    Create a directory if it doesn't exist and
    return the resolved and expanded path.

    Args:
        path (Path | str):
            A directory as a path or a string.

    Returns:
        Path:
            The resolved and expanded path.
    """

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path.expanduser().resolve().absolute()


def timestamp(ms: bool = False) -> tuple[str, str]:
    """
    Create a datestamp and a timestamp as formatted strings.

    Args:
        ms (bool, optional):
            Use millisecond precision. Defaults to False.

    Returns:
        tuple[str, str]:
            A tuple containing:
                1. The formatted date.
                2. The formatted time.
    """

    # Simplified ISO format (no timezone, etc.)
    fmt = "%Y-%m-%d %H-%M-%S"
    end = None

    if ms:
        # Use ms precision
        fmt += ":%f"
        end = -3

    # Return (date, time).
    # UTC time is used to avoid ambiguity.
    return datetime.strftime(datetime.now(datetime.UTC), fmt)[:end].split()


def thread_id() -> int:
    """
    Get the ID of the current thread.

    Returns:
        int:
            The thread ID.
    """
    return threading.get_ident()


def cartesian_prod(
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> np.ndarray:
    """
    Compute the Cartesian product of two 1D arrays.

    Args:
        arr1 (np.ndarray):
            First array.

        arr2 (np.ndarray):
            Second array.

    Returns:
        np.ndarray:
            The resulting Cartesian product.
    """
    return np.transpose([np.repeat(arr1, len(arr2)), np.tile(arr2, len(arr1))])


def load_image(
    path: Path,
    grayscale: bool = False,
    scale: float = None,
) -> np.ndarray:
    """
    _summary_

    Args:
        path (Path): _description_
        grayscale (bool, optional): _description_. Defaults to False.
        scale (float, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """

    # Load the image
    frame = ski.io.imread(path, as_gray=grayscale)

    if scale is not None:
        frame = ski.transform.rescale(
            frame,
            scale,
            channel_axis=2 if not grayscale else None,
        )

    return np.expand_dims(ski.util.img_as_ubyte(frame), axis=0)


def load_video(
    path: Path,
    grayscale: bool = False,
    scale: float = None,
    probe: bool = False,
) -> np.ndarray:
    """
    _summary_

    Args:
        path (Path): _description_
        grayscale (bool, optional): _description_. Defaults to False.
        scale (float, optional): _description_. Defaults to None.
        probe (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """

    av.logging.set_level(av.logging.VERBOSE)
    container = av.open(path)

    v = container.streams.video[0]
    w, h = v.width, v.height

    frames = [None for _ in range(v.frames)]

    for index, frame in tqdm(
        enumerate(container.decode(video=0)),
        total=1 if probe else v.frames,
        desc=f"Loading file {path.name}",
    ):
        if scale is not None:
            frame = frame.reformat(width=int(w * scale), height=int(h * scale))

        frame = frame.to_image()

        if grayscale:
            frame = frame.convert("L")

        frames[index] = ski.util.img_as_ubyte(frame)

    frames = np.array(frames, dtype=np.ubyte)
    return frames


def is_notebook() -> bool:
    """
    Determine if the caller is running in a Jupyter notebook.

    Courtesy of https://stackoverflow.com/a/39662359/4639195.

    Returns:
        bool:
            True if running in a notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        match shell:
            case "ZMQInteractiveShell":
                # Jupyter notebook or qtconsole
                return True
            case "TerminalInteractiveShell":
                # Terminal running IPython
                return False
            case _:
                # Other type (?)
                return False
    except NameError:
        # Probably standard Python interpreter
        return False
