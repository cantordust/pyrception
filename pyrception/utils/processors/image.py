from pathlib import Path
import numpy as np
import skimage as ski

def load_image(
    path: Path,
    greyscale: bool = False,
    scale: float = None,
) -> np.ndarray:
    """
    Load an image file.

    Args:
        path: The path to the image.
        greyscale: If True, the image will be converted to greyscale.
        scale: Scale the image.

    Returns:
        The image as a NumPy array.
    """

    # Load the image
    frame = ski.io.imread(path, as_gray=greyscale)

    if scale is not None:
        frame = ski.transform.rescale(
            frame,
            scale,
            channel_axis=2 if not greyscale else None,
        )

    return np.expand_dims(ski.util.img_as_ubyte(frame), axis=0)
