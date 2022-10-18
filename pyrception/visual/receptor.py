# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Tuple
from typing import Optional

# --------------------------------------
from loguru import logger

# --------------------------------------
import numpy as np

# --------------------------------------
import math


# --------------------------------------
import enum

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from dotmap import DotMap

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf
from torch.distributions.one_hot_categorical import OneHotCategorical

pt.set_printoptions(edgeitems=20)


# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception import conf
from pyrception.visual.util.types import View
from pyrception.visual.util.types import KernelSizeDist
from pyrception.visual.util.types import KernelWeightDist
from pyrception.visual.proto import ProtoLayer


class ReceptorLayer(ProtoLayer):

    """
    A layer of receptors and horizontal cells.

    This layer applies local normalisation to the input
    by looking at an eccentricity-dependent patch of receptors
    and their activations to normalise the activation of the
    receptor in the centre.
    """

    def __init__(
        self,
        original_shape: pt.Tensor,
        scaled_height: Optional[int] = None,
        scaled_width: Optional[int] = None,
        saccades: bool = False,
        mode: Optional[enum.Enum] = cv.COLOR_RGB2GRAY,
        k_min: int = 1,
        k_max: int = 15,
        mh: Optional[int] = None,
        mw: Optional[int] = None,
        sh: int = 1 / 8,
        sw: int = 1 / 8,
        kdist: KernelSizeDist = KernelSizeDist.Gaussian,
        kwdist: KernelWeightDist = KernelWeightDist.Gaussian,
        decreasing: bool = True,
        smooth: bool = True,
    ):

        # Dimensions and resize flag
        self.dim = self._compute_dimensions(
            original_shape,
            scaled_height,
            scaled_width,
            saccades,
        )

        logger.info(f"==[ receptor ] dim: {self.dim}")

        # Create a flatmask
        self.flatmask = self.make_flatmask(mode)

        # Change the depth of the processed frame.
        if self.flatmask is not None:
            self.dim.D = 1

        (self.kmask, ksizes) = self.get_kdist(
            self.dim.padded.H,
            self.dim.padded.W,
            k_min,
            k_max,
            mh,
            mw,
            sh,
            sw,
            kdist,
            decreasing,
            smooth,
        )

        # Create the receptive fields.
        # If saccades are supported,
        # the receptive field map is twice
        # the size of the original input in each dimension.
        self.rf = self._make_rf(
            ksizes,
            kwdist=kwdist,
        )

    @logger.catch
    def make_flatmask(
        self,
        mode: Optional[enum.Enum] = None,
    ):
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        if self.dim.orig.D == 1 or mode is None:
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.dim.H, self.dim.W, self.dim.D))

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = OneHotCategorical(probs)

        return ohc.sample()

        # print(f"==[ R: {self.flatmask[:,:,0].sum()}")
        # print(f"==[ G: {self.flatmask[:,:,1].sum()}")
        # print(f"==[ B: {self.flatmask[:,:,2].sum()}")

        # # Create the actual tensor
        # self.st = pt.sparse_coo_tensor(
        #     np.array(
        #         [
        #             sparse_rows,
        #             sparse_cols,
        #         ]
        #     ),
        #     np.array(sparse_vals),
        #     size=(st_size, st_size),
        # ).to_sparse_csr()

    @logger.catch
    def process(
        self,
        frame: pt.Tensor,
        offset: Optional[Tuple[float, float]] = None,
    ) -> pt.Tensor:
        """
        Read the input by applying a certain offset:
        - Read the next frame
            - (Optional) flatten the frame (remove all channel information)
        - Get the input padding corresponding to the specified offset
        - Compute the local contrast normalisation.
        """

        views = {View.Original: frame}

        # Flatten the frame
        if self.flatmask is not None:
            frame *= self.flatmask

        if offset is not None:
            padding = self._get_padding(offset[0], offset[1])
            padded = self.pad(frame, padding)

        else:
            padded = frame

        mean = self._convolve(padded, self.rf, self.dim.padded.H, self.dim.padded.W)

        if offset is not None:
            mean = self.unpad(mean, padding)

        views[View.ReceptorMean] = mean

        # Subtract the mean and scale
        adapted = frame - mean
        views[View.ReceptorAdapted] = adapted

        return views
