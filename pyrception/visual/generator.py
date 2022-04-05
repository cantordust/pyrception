# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import List
from typing import Tuple
from typing import Optional

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt

# --------------------------------------
import math

# --------------------------------------
import random


class SignalGenerator:
    """
    Signal generator class.
    """

    def __init__(
        self,
        _rows: int,
        _cols: int = 1,
    ):

        """
        Signal generator.

        Produces various signals with the right shape.
        """

        self.rows = _rows
        self.cols = _cols

    def _sparse(
        self,
        _indices: List[List[int]],
        _values: List[float],
        _shape: Optional[Tuple[int]] = None,
    ) -> pt.Tensor:

        return pt.sparse_coo_tensor(
            np.array(_indices),
            np.array(_values),
            (
                self.rows,
                self.cols,
            )
            if _shape is None
            else _shape,
        ).to_sparse_csr()

    def constant(
        self,
        _magnitude: float = 1.0,
    ):

        """
        Constant input.
        """

        return pt.full(
            (self.rows,),
            _magnitude,
        )

    def gnoise(
        self,
        _mean: float = 0.0,
        _sd: float = 1.0,
    ):

        """
        Gaussian noise.
        """

        return pt.normal(
            _mean,
            _sd,
            (self.rows,),
        )

    def unoise(
        self,
        _min: float = -1.0,
        _max: float = 1.0,
    ):

        """
        Uniform noise.
        """

        return pt.FloatTensor(self.rows, self.cols,).uniform_(
            _min,
            _max,
        )

    def flash(
        self,
        _magnitude: float = 1.0,
        _pos: int = 0,
        _size: int = 4,
    ):

        indices = [[i, 0] for i in range(_pos, _pos + _size)]
        values = [_magnitude] * len(indices)

        return pt.reshape(self._sparse(indices, values).to_dense(), (self.rows,))


class GratingGenerator(SignalGenerator):
    """
    A generator for grating patterns.
    """

    def __init__(
        self,
        _rows: int,
        _cols: int = 1,
        _tp: int = 2,  # Temporal period
        _ext: int = 1,  # Spatial extent
        _gap: int = 1,  # Gap size
    ):

        """
        Grating signal generator.
        """

        assert (
            _gap + _ext < _rows
        ), f"==[ Error: Invalid combination of gap and extent (gap + ext must be less than {_rows})"

        super().__init__(_rows, _cols)

        self.grating = pt.zeros(_rows, _cols)

        indices = [[], []]
        values = []
        x = 0

        while x < _rows:
            if x % (_ext + _gap) < _ext:
                indices[0].append(x)
                indices[1].append(0)
                values.append(1.0)
            x += 1

        on_off = self._sparse(
            indices,
            values,
            (_rows, _cols),
        )
        self.grating += on_off.to_dense()

        self.tp = _tp
        self.step = 0

    def roll(
        self,
        _shift: int = 1,
        _axis: int = 0,
    ):

        if self.step >= self.tp:
            self.grating = pt.roll(self.grating, _shift, _axis)
            self.step = 0

        self.step += 1
        return self.grating
