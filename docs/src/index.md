# Overview
Pyrception is a simulation framework for bio-plausible simulation of perceptual modalities. Currently, it supports visual pathways of the mammalian retina, but the long-term goal is to support modalities such as auditory, olfactory and so forth. It can also serve as an input conversion library for encoding raw multimodal sensory input into a uniform spike train suitable for processing with spiking neural networks.

At this stage, only the visual package is implemented. The auditory and olfactory packages are work in progress. Contributions are welcome in case you would like to help with the implementation of these modalities!

## Installation

You can install Pyrception from PyPI:

```shell
pip install pyrception
```

or directly from GitHub (optionally in development mode):

!!! note

    Cloning using `HTTPS` or `SSH`.

    === "HTTPS"

        ``` shell
        git clone https://github.com/cantordust/pyrception.git
        ```

    === "Git+SSH"

        ``` shell
        git clone git@github.com:cantordust/pyrception.git
        ```

``` shell
cd pyrception
```
``` shell
pip install -e .
```

## Usage

Please refer to the [documentation](#documentation), which contains a [step-by-step notebook](notebooks/image.ipynb) demonstrating how to use `pyrception` with a static image. More notebooks are currently being developed, including frame-based RGB input and sparse event input from an event camera. Watch this space.

## Documentation

To generate the documentation, run the MkDocs build pipeline. Note that to build and view the documentation locally, you have to install `pyrception` from GitHub with the optional `docs` modifier:

```shell
pip install -e .[dev]
cd docs
mkdocs build
```

Then, to view the documentation locally, start the MkDocs server:

```shell
mkdocs serve
```

# ToDo

## Short-term
### Visual package
- [X] All major types of retinal cells.
    - [X] Receptors (raw input, Weber's law).
    - [X] Horizontal cells (mean local brightness, normalising feedback).
    - [X] Bipolar cells (positive and negative contrast, temporal filter, excitatory input to ganglion cells).
    - [X] Amacrine cells (inhibitory input to ganglion cells, modulatory signal to bipolar cells).
    - [X] Ganglion cells (spiking).
- [X] Logpolar kernel arrangement.
- [X] Uniform or Gaussian kernels.
- [X] Arbitrary kernel, size, shape and orientation.
- [ ] Saccadic movements [WIP].
- [ ] Colour vision (with colour opponency) [WIP].
- [ ] Temporal dynamics [WIP].
- [ ] Events as input [WIP].

### Auditory package
WIP.

### Olfactory package
WIP.

### Overall functionality
- [WIP] Support alternative backends for sparse matrix operations ([CuPy](https://cupy.dev/), [PyTorch](https://pytorch.org/docs/stable/sparse.html), [Sparse](https://sparse.pydata.org/en/stable/)).
- [ ] Interfacing with (neuromorphic) hardware, such as event cameras.
