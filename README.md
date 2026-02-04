[![PyPI - Version](https://img.shields.io/pypi/v/pyrception)](https://pypi.org/project/pyrception/)
[![Read The Docs](https://readthedocs.org/projects/pyrception/badge/?version=latest)](https://pyrception.readthedocs.io/en/latest/)

# 🌄 Overview
Pyrception is a simulation framework for biosensors. Currently, it provides the base ingredients for simulating key parts of the structural and functional elements of visual processing observed in the mammalian retina. The long-term goal of Pyrception is to support multiple sensory modalities (such as auditory, olfactory and tactile), and to provide methods for integrating those inputs into a unified multisensory input signal (such as spike trains). Alongside this, Pyrception can also serve as an input *conversion*  for encoding raw multimodal sensory input into a uniform spike train suitable for processing with spiking neural networks.

<!-- ![Herd of zebra - multi-stage retinal processing]() -->
<div align="center">
    <video align="center" src="https://github.com/cantordust/pyrception/raw/refs/heads/dev/docs/src/notebooks/__temp__.mp4" type="video/mp4" controls preload autoplay muted loop width="800" ></video>
    <p align="center">An RGB video processed with different layers (see the full <a href="https://pyrception.readthedocs.io/en/latest/notebooks/video/">example</a>).</p>
</div>

## 🪛 Installation

You can install Pyrception from PyPI:

```shell
pip install pyrception
```

or directly from GitHub (optionally in development mode):

```shell
git clone git@github.com:cantordust/pyrception.git
cd pyrception
pip install -e .
```

### ♻️ Optional dependencies

Pyrception supports several dependency groups:

- `cli`: Command-line interface.
- `events`: Support for processing events (including from event cameras).
- `dev`: Development tools (for testing, profiling, etc.).
- `torch`: PyTorch support.
- `ipy`: ipykernel & ipywidgets (for running inside notebooks).
- `docs`: Tools for building the documentation.
    - NOTE: The documentation is built using [MKDocs](https://www.mkdocs.org/), which has been discontinued. The documentation will likely move to [Zensical](https://zensical.org/) soon.
- `all`: All of the above.

Use the `--group` with `pip` to enable a dependency group (repeat for each group). For instance:

```shell
pip install -e . --group events --group docs
```

will pull in all dependencies necessary for event-based input and building the documentation.

## ⏯️ Usage

Please refer to the [documentation](https://pyrception.readthedocs.io/en/latest/), which contains step-by-step notebooks demonstrating how to use Pyrception with a [static image](https://pyrception.readthedocs.io/en/latest/notebooks/image/) and an [RGB video](https://pyrception.readthedocs.io/en/latest/notebooks/video/). More notebooks are currently being developed, including sparse event input from an event camera. Stay tuned.

## 📈 Development

Please open an issue if you discover a bug, feature. That said, contributions are welcome!

To generate and view the documentation locally, clone the repository and run the MkDocs build pipeline (note that you have to install Pyrception with the [`docs` dependency group](#optional-dependencies)):

```shell
git clone git@github.com:cantordust/pyrception.git
cd pyrception
pip install -e . --group docs
cd docs
mkdocs build
```

Then, to view the documentation locally, start the MkDocs server:

```shell
mkdocs serve
```

# 📋 ToDo

## Short-term
### 👁️ Visual package
- [X] All major types of retinal cells.
    - [X] Receptors (raw input, Weber's law).
    - [X] Horizontal cells (mean local brightness, normalising feedback).
    - [X] Bipolar cells (positive and negative contrast, temporal filter, excitatory input to ganglion cells).
    - [X] Amacrine cells (inhibitory input to ganglion cells, modulatory signal to bipolar cells).
    - [X] Ganglion cells (spiking).
- [X] Logpolar kernel arrangement.
- [X] Uniform or Gaussian kernels.
- [X] Arbitrary kernel, size, shape and orientation.
- 🚧 Colour vision (with colour opponency).
- 🚧 Temporal dynamics.
- 🚧 Events as input.
- [ ] Saccadic movements.

### 👂 Auditory package
WIP.

### 👃 Olfactory package
WIP.

### 🔧 Others
- 🚧 Support alternative backends for sparse matrix operations ([CuPy](https://cupy.dev/), [PyTorch](https://pytorch.org/docs/stable/sparse.html), [Sparse](https://sparse.pydata.org/en/stable/)).
- 🚧 Interfacing with (neuromorphic) hardware, such as event cameras.
