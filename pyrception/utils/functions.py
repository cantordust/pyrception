from pathlib import Path
from datetime import datetime, UTC
from collections.abc import Iterable
from matplotlib import colors

import numpy as np
from IPython import get_ipython


def mkdir(path: Path | str) -> Path:
    """
    Create a directory if it doesn't exist and
    return the resolved and expanded path.

    Args:
        path: A directory as a path or a string.

    Returns:
        The resolved and expanded path.
    """

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path.expanduser().resolve().absolute()


def timestamp(
    ms: bool = False,
    split: bool = False,
) -> str | tuple[str, str]:
    """
    Create a datestamp and a timestamp as formatted strings.

    Args:
        ms: Use millisecond precision.
        split: Split the string into date and time components.

    Returns:
        The formatted date and time.
    """

    # Simplified ISO format (no timezone, etc.)
    fmt = "%Y-%m-%d_%H-%M-%S"
    end = None

    if ms:
        # Use ms precision
        fmt += "-%f"
        end = -3

    # Return (date, time).
    # UTC time is used to avoid ambiguity.
    ts = datetime.strftime(datetime.now(UTC), fmt)[:end]
    return ts.split("_") if split else ts


def to_rgba(
    colour: str | Iterable,
) -> np.ndarray:
    """
    Convert a colour specified as HEX into RGBA (a 4-tuple).

    Args:
        colour: The colour specified as a HEX string or an iterable.

    Returns:
        The RGBA values of the colour.
    """
    if colour is not None:
        if isinstance(colour, str):
            colour = colors.to_rgba(colour)
        colour = list(colour)
        if len(colour) == 3:
            colour.append(1.0)
        colour = np.array(colour)

    return colour


def cartesian_prod(
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> np.ndarray:
    """
    Compute the Cartesian product of two 1D arrays.

    Args:
        arr1: The first array.
        arr2: The second array.

    Returns:
        The resulting Cartesian product.
    """

    mg = np.meshgrid(arr1, arr2, indexing="ij")
    return np.concatenate(np.stack((mg[0], mg[1]), axis=2), axis=0)


def arg2np(
    arg: None | int | float | tuple | np.ndarray,
    ext: int = 4,
    pad: int = 0,
    val: int = 0,
    fill: bool = False,
    bounds: tuple[int | float, ...] | None = None,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    """
    Convert an argument into a NumPy array, assuming that we
    want the values to apply to some 2D property, such as image
    or kernel size, field of view, etc.

    Args:
        arg: The argument.
        ext: Extent of the returned array.
        pad: Pad the array to the `pad` size.
        val: Value to use to pad the array.
        fill: If `True` and `arg` is a number, fill the array up to a size of
            `ext` with the value of `arg`.
        bounds: Upper and lower bounds to apply to the argument.
        dtype: The dtype to use when converting to NumPy.

    Raises:
        AttributeError:
            Raised if the parameter is a tuple with 3 or more than 4 elements.

        AttributeError:
            Raised if the parameter is a NumPy array with 3 or more than 4 elements.

    Returns:
        The argument as a NumPy array.
    """

    ext = np.clip(ext, 0, None)
    pad = np.clip(pad, 0, None)

    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise TypeError(
                f"Invalid bounds: '{bounds}' (must be either None or a tuple of length 2)."
            )

    if arg is None:
        arg = np.full((ext,), val)

    elif isinstance(arg, (int, float)):
        arg = np.full((ext,), arg) if fill else np.array([arg])

    elif isinstance(arg, (tuple, list)):
        arg = np.array(arg)

    # At this point, only proceed if we have a NumPy array
    if not isinstance(arg, np.ndarray):
        raise TypeError(f"Invalid argument type: '{type(arg)}'")

    arg = arg.astype(dtype)
    if arg.size < pad:
        arg = np.concatenate((arg, np.full((pad - arg.size,), val, dtype=dtype)))

    if bounds is not None:
        arg = np.clip(arg, a_min=bounds[0], a_max=bounds[1])

    return arg[:ext]


def make_substrate(
    height: int,
    width: int,
    step: int = 1,
) -> np.ndarray:
    """
    Create a Cartesian coordinate mesh with all possible combinations
    of widths and heights (= columns and rows). These are the coordinates
    of all the pixels in the raw input.

    Args:
        height: Substrate height.
        width: Substrate width.
        step: Grid step (defines the coarseness of the mesh).

    Returns:
        A substrate with the specified coarseness.
    """

    # A mesh of all possible coordinate pairs
    rows = np.linspace(0, height - 1, height // step, dtype=np.int32)
    cols = np.linspace(0, width - 1, width // step, dtype=np.int32)
    return cartesian_prod(rows, cols)


def crop_to_fov(
    coordinates: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Crop some substrate coordinates to the
    dimensions of the visual field (the field of view, or FoV).

    Args:
        coordinates: Substrate coordinates (rows, cols).
        size: The size of the FoV.

    Returns:
        The FOV mask and the index array for the subset of
        unique coordinates cropped to the visual field.
    """

    coordinates = coordinates.astype(np.int32)

    # Crop the coordinates to the FoV.
    fov_mask = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < size[0])
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < size[1])
    )

    cropped = coordinates[fov_mask]

    # Indices of the unique coordinates.
    unique_indices = np.unique(cropped, return_index=True, axis=0)[1]

    return (cropped, fov_mask, unique_indices)


def is_notebook() -> bool:
    """
    Determine if the caller is running in a Jupyter notebook.

    Credit: https://stackoverflow.com/a/39662359/4639195.

    Returns:
        bool: True if running in a notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        match shell:
            case "ZMQInteractiveShell":
                # Jupyter notebook or qtconsole
                return True
            case _:
                # Other type (?)
                return False
    except NameError:
        # Probably standard Python interpreter
        return False
