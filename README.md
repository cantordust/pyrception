# Overview
Pyrception provides an easy way to interface various types of input data with neural networks.

## Installation
Install Pyrception as a local package

`$ pip install --user pyrception`

**Or** if you would like to be able to modify the source:

`$ pip install --user -e .`

If you are in a virtual environment, you can skip the `--user` part. All requirements should be installed automatically.

# ToDo

## Short-term
- Receptor adaptation following Weber's law
- Proper saccadian movements
- Retinal ganglion cells

# Example
A very simple example using visual module:

```python

import cv2 as cv
from pyrception.visual.util.types import KernelDist
from pyrception.visual.retina import Retina


def test():

    src = cv.VideoCapture(0)

    saccades = False

    retina = Retina(
        src,
        scaled_height=256,
        saccades=saccades,
        receptor_args={
            "kdist": KernelDist.Gaussian,
            "sh": 1 / 16,
            "sw": 1 / 16,
        },
        bipolar_args={
            "alpha": 0.1,
            "kdist": KernelDist.Gaussian,
        },
    )

    retina.run(saccades=saccades)


if __name__ == "__main__":
    test()

```