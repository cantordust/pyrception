from typing import *

# --------------------------------------
from loguru import logger

# --------------------------------------
import enum

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt
from torch.distributions import OneHotCategorical

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception.visual.util.types import View
from pyrception.visual.util.types import Dim
from pyrception.visual.util.types import Dims
from pyrception.visual.layers.base import BaseLayer


class ReceptorLayer(BaseLayer):
    """
    A layer of receptors.

    This layer applies the first processing steps to the raw input.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        height: int = None,
        width: int = None,
        saccades: bool = False,
        mode: Optional[enum.Enum] = cv.COLOR_RGB2GRAY,
        name: str = "Receptor",
    ):

        name = f"{name:<10s}"
        super().__init__(name)

        self.info("Initialising...")

        # Dimensions and resize flag
        self.dims = self._compute_dimensions(
            shape,
            height,
            width,
            saccades,
        )

        # Create a flatmask
        self.flatmask = self.make_flatmask(mode)

        # Change the depth of the processed frame.
        if self.flatmask is not None:
            self.dims.comp.depth = 1

        self.info("Initialised.")

    # @logger.catch
    def make_flatmask(
        self,
        mode: Optional[enum.Enum] = None,
    ):
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        if self.dims.orig.depth == 1 or mode is None:
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.dims.H, self.dims.W, self.dims.D))

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

    # @logger.catch
    def process(
        self,
        frame: pt.Tensor,
        offset: Optional[Tuple[float, float]],
        views: Dict[View, pt.Tensor],
        n_frame: int,
        save_frames: Set[int],
        save_views: Set[View],
        frame_paths: Optional[Dict[View, Path]],
    ) -> pt.Tensor:
        """
        Read the input by applying a certain offset:

        NOTE: The following steps need to be updated.
        # - Read the next frame
        # - (Optional) flatten the frame (remove all channel information)
        # - Get the input padding corresponding to the specified offset (only if saccades are active)
        # - Compute the local contrast normalisation (horizontal cell effect).
        """

        # _views = {View.Original: frame}

        # # Flatten the frame
        # if self.flatmask is not None:
        #     frame *= self.flatmask

        # This is where saccades happen!
        if offset is not None:
            padding = self._get_padding(offset[0], offset[1])
            padded = self.pad(frame, padding)

        else:
            padded = frame

        # Convolve the input with the kernel matrix
        mean = self._convolve(padded, self.rf, self.dims.padded.H, self.dims.padded.W)

        if offset is not None:
            mean = self.unpad(mean, padding)

        _views[View.ReceptorMean] = mean

        # Subtract the mean and scale
        adapted = frame - mean
        _views[View.ReceptorAdapted] = adapted

        if n_frame in save_frames:
            self._save_views(_views, n_frame, save_views, frame_paths)

        views.update(_views)
