from typing import Union

# --------------------------------------
from pathlib import Path

# --------------------------------------
from pyrception.util.functions import cwd
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
            "rfsizedist": RFSizeDist.Flat,
            "rftype": KernelType.Flat,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 3,
            "k_max": 4,
        },
        bipolar_args={
            "alpha": 0.9,
            "rfsizedist": RFSizeDist.Flat,
            "rftype": KernelType.Flat,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 3,
            "k_max": 3,
        },
        ganglion_args={
            "rfsizedist": RFSizeDist.Flat,
            "rftype": KernelType.Flat,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 3,
            "k_max": 3,
        },
    )

    retina.run(
        saccades,
        output_path=cwd(__file__) / "output",
        save_video=False,
        save_frames={1, 2, 3},
    )


if __name__ == "__main__":

    # * Camera * #
    source = 0

    # * Run the example * #
    run(source)
