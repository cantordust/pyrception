from typing import *

# --------------------------------------
import torch as pt
from torch.nn import functional as ptf
from torch.distributions import OneHotCategorical

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
import cv2 as cv

# --------------------------------------
from pyrception import conf
from pyrception.visual import Dim
from pyrception.visual import Dims
from pyrception.visual.layers.base import BaseLayer
from pyrception.visual.rf import ReceptiveFields


class ReceptorLayer(BaseLayer):
    """
    A layer of receptors.

    This layer applies the first processing steps to the raw input,
    including saccades and RGB to grayscale conversion.

    TODO
    Fix the flatmask so that we can use 3D inputs.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        saccades: bool = False,
        mode: Optional[int] = cv.COLOR_RGB2GRAY,
        name: str = "Receptor",
    ):

        # Initialise the base
        super().__init__(size, name)

        self.saccades = saccades
        self.mode = mode  # TODO: Connect this to the dimensionality of the input.

        # Dimensionality of the input.
        # The padded version is used for saccades.
        self.dims = self._compute_dimensions()

        # Create a flatmask
        self.flatmask = self._make_flatmask(mode)

        # Receptive fields
        self.rfs = ReceptiveFields(
            self.size,
            name=f"{name} RFs",
        )

        # Receptor activation
        self.activation = None

        self.info("Initialised.")

    def _compute_dimensions(self) -> Dims:
        """
        Compute the dimensions for the input, with optional padding
        if saccades are enabled.

        Returns:
            Dims:
                Dimensions of the visual field, optionally padded.
        """
        # This is just for convenience
        (height, width, depth) = self.size

        dims = Dims(Dim(height, width, depth))

        # Left and right padding
        lr_padding = (width // 2 + width % 2) if self.saccades else 0

        # Top and bottom padding
        tb_padding = (height // 2 + height % 2) if self.saccades else 0

        dims.padded.width = width + 2 * lr_padding
        dims.padded.height = height * depth + 2 * tb_padding
        dims.padded.depth = depth
        dims.padded.span = dims.padded.width * dims.padded.height * dims.padded.depth

        # The frame is padded only at the top and the bottom,
        # but the left and right padding values are used
        # to compute the size of the retinal field below.
        dims.padding = np.array([lr_padding, lr_padding, tb_padding, tb_padding])

        return dims

    def _pad(
        self,
        frame: pt.Tensor,
        padding: Tuple[int, int, int, int],
    ) -> pt.Tensor:
        """
        Pad the frame so that we can shift the FOV
        without making the frame 'jump'.
        This is part of the implementation of saccadic movements.

        Args:
            frame (pt.Tensor):
                Frame to be padded.

            padding (Tuple[int, int, int, int]):
                Padding extents in PyTorch order (left, right, top, bottom).

        Returns:
            pt.Tensor:
                The padded frame.
        """

        padded = ptf.pad(frame, padding)

        return padded

    def _unpad(
        self,
        tensor: pt.Tensor,
        padding: Tuple[int, int, int, int],
    ) -> pt.Tensor:
        """
        Unpad a frame padded with _pad().

        Args:
            tensor (pt.Tensor):
                Padded tensor.

            padding (Tuple[int, int, int, int]):
                Amount of padding in PyTorch order (left, right, top, bottom).

        Returns:
            pt.Tensor:
                Unpadded tensor.
        """

        return tensor[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

    def _as_image(
        self,
        tensor: pt.Tensor,
    ) -> pt.Tensor:
        """
        Convert a tensor to an 8-bit image frame.

        Args:
            tensor (pt.Tensor):
                Frame to be converted into an 8-bit integer NumPy array.

        Returns:
            pt.Tensor:
                The 8-bit frame.
        """

        tmin = tensor.min()

        return (255 * (tensor - tmin) / (tensor.max() - tmin + 1e-8)).type(pt.uint8)

    def _scale(
        self,
        tensor: pt.Tensor,
        min: Optional[float] = 0.0,
        max: Optional[float] = 255.0,
    ) -> pt.Tensor:
        """
        Min-max normalised version of the frame.

        Args:
            tensor (pt.Tensor):
                The tensor to be normalised.

            min (Optional[float], optional):
                Minimal value. Defaults to 0.0.

            max (Optional[float], optional):
                Maximal value. Defaults to 255.0.

        Returns:
            pt.Tensor:
                The normalised tensor.
        """

        tmin = tensor.min()
        tmax = tensor.max()

        return min + (max - min) * (tensor - tmin) / (tmax - tmin)

    def _stretch(
        self,
        tensor: pt.Tensor,
    ) -> pt.Tensor:
        """
        Stretch a 2D or 3D input (image) into a 1D vector.

        Args:
            tensor (pt.Tensor):
                The tensor to flatten.

        Returns:
            pt.Tensor:
                The flattened tensor
        """

        # TODO: 3D -> 2D
        # TODO: Colour opponency?
        # TODO: Handle transparency (4D tensors)?
        # if frame.dim() == 3:
        #     # Transpose the depth dimension and stretch
        #     return frame.permute(2, 1, 0).flatten()[:, None]

        return tensor.flatten()

    def _fold(
        self,
        tensor: pt.Tensor,
        height: int,
        width: int,
    ) -> pt.Tensor:
        """
        Fold a 1D vector into a 2D tensor.

        Args:
            tensor (pt.Tensor):
                Tensor to be folded.

            height (int):
                Height of the resulting tensor.

            width (int):
                Width of the resulting tensor.

        Returns:
            pt.Tensor:
                The folded tensor.
        """

        return tensor.reshape(width, height)

    def _get_padding(
        self,
        height_offset: float = 0.0,
        width_offset: float = 0.0,
    ) -> Tuple[int, ...]:
        """
        Compute the horizontal and vertical offsets
        from the width and height offset values and
        then compute the padding values from the offsets.

        Args:
            height_offset (float, optional):
                Offset to move in the height direction. Defaults to 0.0.

            width_offset (float, optional):
                Offset to move in the width direction. Defaults to 0.0.

        Returns:
            Tuple[int, ...]:
                A tuple containing the padding in PyTorch order (left, right, top, bottom).
        """

        # TODO
        # Boundary checks.

        # wpo: width-wise pixel offset
        # hpo: height-wise pixel offset
        wpo = int(
            np.sign(width_offset)
            * math.floor(math.fabs(width_offset) * self.dims.original.width)
        )
        hpo = int(
            np.sign(height_offset)
            * math.floor(math.fabs(height_offset) * self.dims.original.height)
        )

        padding = tuple(self.dims.padding + np.array([hpo, -hpo, wpo, -wpo]).tolist())

        return padding

    def _make_flatmask(
        self,
        mode: Optional[int] = None,
    ) -> pt.Tensor:
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.

        WIP: Needs to be tested.

        Args:
            mode (Optional[int], optional):
                Masking mode. Defaults to None.

        Returns:
            pt.Tensor:
                The mask to be applied to the raw input.
        """

        if self.dims.original.depth == 1 or mode is None:
            return

        # print(f"==[ Creating flatmask...")

        probs = pt.zeros(self.size, dtype=conf.dtype)

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = OneHotCategorical(probs)

        return ohc.sample()

    def forward(
        self,
        frame: pt.Tensor,
        offset: Optional[Tuple[float, float]] = None,
    ) -> pt.Tensor:
        """
        Read the input and apply a certain offset if saccades are enabled.

        Args:
            frame (pt.Tensor):
                The raw input.

            offset (Optional[Tuple[float, float]], optional):
                Padding for saccades. Defaults to None.

        Returns:
            pt.Tensor:
                The frame with optional padding (for saccades).
        """

        if offset is not None:
            # This is where saccades happen!

            # WIP
            # This needs to be revised because the frame
            # needs to be unrolled properly.
            # ==================================================
            padding = self._get_padding(offset[0], offset[1])
            frame = ptf.pad(frame, padding)

            patch = frame[
                padding[2] : -padding[3],
                padding[0] : -padding[1],
            ]

        self.activation = frame.flatten()

        return self.activation
