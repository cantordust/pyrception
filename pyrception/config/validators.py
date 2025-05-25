import numpy as np

from typing import Iterable


def array_validator(
    value: Iterable,
    min_len: int = 0,
    max_len: int = 2,
    fixed_len: int = None,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    """
    Validate an array

    Args:
        value (Any):
            The value to be validated.

        min_len (int, optional):
            The minimal length. Defaults to 2.

        max_len (int, optional):
            The maximal length. Defaults to 2.

        dtype (np.dtype, optional):
            The data type for the resulting array. Defaults to np.int32.

    Raises:
        TypeError:
            Raised if the value is not iterable.

        ValueError:
            Raised if the value doesn't conform to the size constraints.

    Returns:
        np.ndarray:
            Return a NumPy array.
    """
    if not isinstance(value, (tuple, list, np.ndarray, int, float)):
        raise TypeError(f"Validation error: value {value} is not iterable.")
    if len(value) < min_len:
        raise ValueError(f"Validation error: {value} must have a length >= {min_len}.")
    if len(value) > max_len:
        raise ValueError(f"Validation error: {value} must have a length <= {max_len}.")
    if isinstance(value, (int, float)) and fixed_len is not None:
        value = [value for _ in range(fixed_len)]
    return np.array(value, dtype=dtype)
