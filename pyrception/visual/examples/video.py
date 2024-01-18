from typing import Union

# --------------------------------------
from pathlib import Path

# --------------------------------------
from pyrception.util.functions import curdir
from pyrception.visual.util.types import RFSizeDist
from pyrception.visual.util.types import KernelType
from pyrception.visual.retina import Retina


def run(source: Union[str, int]):

    saccades = False

    retina = Retina(
        source,
        scaled_height=256,
        saccades=saccades,
        receptor_args={
            "rfsizedist": RFSizeDist.Gaussian,
            "rftype": KernelType.Proportional,
            "sh": 0.33,
            "sw": 0.33,
            "k_min": 1,
            "k_max": 4,
        },
        bipolar_args={
            "alpha": 0.95,
            "rfsizedist": RFSizeDist.Gaussian,
            "rftype": KernelType.Proportional,
            "sh": 0.33,
            "sw": 0.33,
            "k_min": 1,
            "k_max": 4,
        },
        ganglion_args={
            "rfsizedist": RFSizeDist.Gaussian,
            "rftype": KernelType.Proportional,
            "sh": 0.33,
            "sw": 0.33,
            "k_min": 1,
            "k_max": 4,
        },
    )

    retina.run(
        saccades,
        save_video=True,
        save_frames=False,
        output_path=curdir(__file__) / "output",
    )


if __name__ == "__main__":

    # * Video file * #
    # Path to the directory containing all sources
    src_path = curdir(__file__) / "sources"
    file = "landing.mp4"
    source = str(src_path / file)

    # * Run the example * #
    run(source)
