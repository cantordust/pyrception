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
            "rfsizedist": RFSizeDist.Flat,
            "rftype": KernelType.Proportional,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 3,
            "k_max": 4,
        },
        bipolar_args={
            "alpha": 0.9,
            "rfsizedist": RFSizeDist.Gaussian,
            "rftype": KernelType.Proportional,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 3,
            "k_max": 5,
        },
        ganglion_args={
            "rfsizedist": RFSizeDist.Gaussian,
            "rftype": KernelType.Proportional,
            "sh": 0.5,
            "sw": 0.5,
            "k_min": 1,
            "k_max": 3,
        },
    )

    retina.run(
        saccades,
        save_video=False,
        save_frames=set(range(50)),
        output_path=curdir(__file__) / "output",
    )


if __name__ == "__main__":

    # * Directory containing images * #
    # Path to the directory containing all sources
    src_path = curdir(__file__) / "sources"
    dir = "MPI-Sintel-complete/test/clean/cave_3"
    source = str(src_path / dir)

    print(f"==[ source: {source}")

    # * Run the example * #
    run(source)
