# Overview
Pyrception aims to serve as a simulation and conversion framework for different perceptual modalities (visual, auditory, olfactory, etc.). Although it is not the primary objective, it can also serve as an input conversion library for encoding raw multimodal sensory input into a uniform spike train suitable for processing with spiking neural networks.

At this stage, only the visual package is implemented. The auditory and olfactory packages are work in progress. If you think you can help with implementation, please reach out - contributions are welcome!

## Installation
Currently, Pyrception is not released as a PyPI or Conda package. You can install Pyrception as a local package in development mode by cloning the Git repository and creating a Conda environment using the provided `environment.yml` file.

`$> conda env create -f environment.yml`

# ToDo

## Short-term
Visual package:
- [X] Receptor signal scaling following Weber's law
- [X] Retinal ganglion cells
- [ ] Saccadic movements (WIP)
- [ ] Colour vision (with colour opponency)

## Mid-term
- [ ] Auditory package (WIP)
- [ ] Olfactory package (WIP)
- [ ] Investigate alternative backends for sparse matrix operations (e.g., CuPy, Trilinos, etc.)