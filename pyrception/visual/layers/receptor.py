import math

import numpy as np

import typing as tp

from pyrception import conf
from pyrception.visual import Dim
from pyrception.visual import Dims
from pyrception.visual.rf import ReceptiveFields
from pyrception.visual.layers.base import BaseLayer


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
        shape: tuple[int, ...],
        scale: float = 1.0,
        saccades: bool = False,
        greyscale: bool = True,
        log_response: bool = True,
        name: str = "Receptor",
        notifier: tp.Callable = None,
    ):
        # Initialise the base
        super().__init__(shape, name, notifier)

        self.saccades = saccades

        # TODO: Connect this to the dimensionality of the input.
        self.greyscale = greyscale

        # Whether the receptors should show a logarithmic response behaviour.
        self.log_response = log_response

        # Dimensionality of the input.
        # The padded version is used for saccades.
        self.dims = self._compute_dimensions()

        # Create a flatmask
        self.flatmask = self._make_flatmask()

        # Receptive fields
        self.rfs = ReceptiveFields(
            self.shape,
            name=f"{name} RFs",
            notifier=notifier,
        )

        self.rfs.cell_coordinates = self.rfs.substrate

        # Receptor activation
        self.membrane = None

        self.info("Initialised.")

    def _compute_dimensions(self):
        """
        Compute the dimensions for the input, with optional padding
        if saccades are enabled.

        Returns:
            Dims:
                Dimensions of the visual field, optionally padded.
        """
        # This is just for convenience
        (height, width, depth) = self.shape

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
        # to compute the shape of the retinal field below.
        dims.padding = np.array([lr_padding, lr_padding, tb_padding, tb_padding])

        return dims

    def _shift(
        self,
        frame: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Pad the frame so that we can shift the FOV
        without making the frame 'jump'.
        This is part of the implementation of saccadic movements.

        Args:
            frame (np.ndarray):
                Frame to be padded.

            padding (tuple[int, int, int, int]):
                Padding extents in the following order: (left, right, top, bottom).

        Returns:
            np.ndarray:
                The padded frame.
        """

        padded = np.pad(frame, padding)

        return padded

    def _unpad(
        self,
        tensor: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Unpad a frame padded with _pad().

        Args:
            tensor (np.ndarray):
                Padded tensor.

            padding (tuple[int, int, int, int]):
                Padding extents in the following order: (left, right, top, bottom).

        Returns:
            np.ndarray:
                Unpadded tensor.
        """

        return tensor[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

    def _as_image(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a tensor to an 8-bit image frame.

        Args:
            tensor (np.ndarray):
                Frame to be converted into an 8-bit integer NumPy array.

        Returns:
            np.ndarray:
                The 8-bit frame.
        """

        tmin = frame.min()

        return (255 * (frame - tmin) / (frame.max() - tmin + 1e-8)).astype(np.uint8)

    def _scale(
        self,
        tensor: np.ndarray,
        min: tp.Optional[float] = 0.0,
        max: tp.Optional[float] = 255.0,
    ) -> np.ndarray:
        """
        Min-max normalised version of the frame.

        Args:
            tensor (np.ndarray):
                The tensor to be normalised.

            min (tp.Optional[float], optional):
                Minimal value. Defaults to 0.0.

            max (tp.Optional[float], optional):
                Maximal value. Defaults to 255.0.

        Returns:
            np.ndarray:
                The normalised tensor.
        """

        tmin = tensor.min()
        tmax = tensor.max()

        return min + (max - min) * (tensor - tmin) / (tmax - tmin)

    def _stretch(
        self,
        tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Stretch a 2D or 3D input (image) into a 1D vector.

        Args:
            tensor (np.ndarray):
                The tensor to flatten.

        Returns:
            np.ndarray:
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
        tensor: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Fold a 1D vector into a 2D tensor.

        Args:
            tensor (np.ndarray):
                Tensor to be folded.

            height (int):
                Height of the resulting tensor.

            width (int):
                Width of the resulting tensor.

        Returns:
            np.ndarray:
                The folded tensor.
        """

        return tensor.reshape(width, height)

    def _shift(
        self,
        frame: np.ndarray,
        height_offset: float = 0.0,
        width_offset: float = 0.0,
    ) -> tuple[int, ...]:
        """
        Computes the horizontal and vertical offsets
        from the width and height offset values,
        then computes the padding values from the offsets
        and shifts the frame by that amount.

        Args:
            frame (np.ndarray):
                The frame to be shifted.

            height_offset (float, optional):
                Offset to move in the height direction. Defaults to 0.0.

            width_offset (float, optional):
                Offset to move in the width direction. Defaults to 0.0.

        Returns:
            np.ndarray:
                The patch corresponding to the shifted frame.
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

        # Padding extents in the following order: (left, right, top, bottom).
        padding = tuple(self.dims.padding + np.array([hpo, -hpo, wpo, -wpo]).tolist())

        frame = np.pad(frame, padding)

        patch = frame[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

        return patch

    def _make_flatmask(self) -> np.ndarray:
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.

        WIP: Needs to be tested.

        Returns:
            np.ndarray:
                The mask to be applied to the raw input.
        """

        if self.dims.original.depth == 1 or self.greyscale:
            return

        self.debug("Creating flatmask...")

        probs = np.zeros(self.shape, dtype=conf.num)

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = np.random.choice(self.shape, n, p=r_prob)

        return ohc.sample()

    def forward(
        self,
        frame: np.ndarray,
        offset: tuple[float, float] | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Read the input and apply a certain offset if saccades are enabled.

        Args:
            frame (np.ndarray):
                The raw input.

            offset (tuple[float, float] | None, optional):
                Padding for saccades. Defaults to None.

            dt (float | None, optional):
                In the case of temporal integration,
                indicates the time since the last input.
                Defaults to None.

        Returns:
            tuple[np.ndarray, float]:
                The frame with optional padding (for saccades).
        """

        # Saccades
        if offset is not None:
            frame = self._shift(frame, offset[0], offset[1])

        # Log response
        if self.log_response:
            frame = np.log1p(frame)

        # Activation
        self.membrane = frame.flatten()

        return self.membrane
