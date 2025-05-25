import numpy as np
from pydantic import field_validator

# import torch as pt
from pyrception.config.base import ConfBase


class NumPyConf(ConfBase):
    dtype: np.dtype = np.dtype(np.float32)

    @field_validator("dtype", mode="before")
    @classmethod
    def str_to_dtype(cls, value: str) -> np.dtype:
        if isinstance(value, np.dtype):
            return value
        return np.dtype(value)


class PyTorchConf(ConfBase):
    # TODO: Add options, e.g., dtype
    # dtype: pt.dtype = pt.dtype(torch.float32)
    pass


class NumConf(ConfBase):
    numpy: NumPyConf = NumPyConf()
    torch: PyTorchConf = PyTorchConf()
