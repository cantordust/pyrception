# Overview
Pyrception is a simulation framework for biorealistic simulation of perceptual modalities. Currently, it supports visual pathways of the mammalian retina, but the long-term goal is to support modalities such as auditory, olfactory and so forth. It can also serve as an input conversion library for encoding raw multimodal sensory input into a uniform spike train suitable for processing with spiking neural networks.

Contributions are welcome, especially for the implementation of modalities other than vision.

## Installation

You can install Pyrception from PyPI, or directly from GitHub.

### PyPI

```shell
pip install pyrception
```

### GitHub

Clone the repository and install it (optionally in in development mode):

``` shell
git clone git@github.com:cantordust/pyrception.git
cd pyrception
pip install -e .
```

### Documentation

To generate the documentation, run the MkDocs build pipeline:

```shell
mkdocs build
```

To view the documentation locally, start the MkDocs server:

```shell
mkdocs serve
```

# ToDo

## Short-term
Visual package:
- [X] Receptor signal scaling following Weber's law.
- [X] Retinal ganglion cells.
- [ ] Saccadic movements (WIP).
- [ ] Colour vision (with colour opponency).
- [ ] Auditory package (WIP).
- [ ] Olfactory package (WIP).
- [ ] Investigate alternative backends for sparse matrix operations ([CuPy](https://cupy.dev/), [PyTorch](https://pytorch.org/docs/stable/sparse.html), [Sparse](https://sparse.pydata.org/en/stable/)).
- [ ] Interfacing with neuromorphic hardware.