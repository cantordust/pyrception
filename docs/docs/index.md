# Overview
Pyrception aims to serve as a simulation and conversion framework for different perceptual modalities. Currently, it supports simulation of visual pathways of the mammalian retina, but the goal is to incorporate other modalities as well, such as auditory, olfactory and so forth. It can also serve as an input conversion library for encoding raw multimodal sensory input into a uniform spike train suitable for processing with spiking neural networks.

At this stage, only the visual package is implemented. The auditory and olfactory packages are work in progress. Contributions are welcome in case you would like to help with the implementation of these modalities!

## Installation

You can install Pyrception from PyPI, or directly from GitHub.

### PyPI

```shell
pip install pyrception
```

### GitHub

Clone the repository and install it (optionally in in development mode).

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