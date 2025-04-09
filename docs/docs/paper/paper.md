---
title: 'Pyrception: A Python package for biologically plausible neuromorphic perception'
tags:
  - Python
  - neuromorphic
  - retinomorphic
  - retina
  - neuroscience
authors:
  - name: Alexander Hadjiivanov
    orcid: 0000-0002-7045-1005
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Giulia D'Angelo
    affiliation: 3
affiliations:
 - name: Advanced Concepts Team, European Space Agency
   index: 1
   ror: 03wd9za21
 - name: Adapsent
   index: 2
 - name: Czech Technical University in Prague
   index: 3
   ror: 03kqpb082
date: 2 Mar 2025
bibliography: references.bib
---

# Summary

Neuromorphic perception is enjoying a resurgence in attention from the scientific community
<<<<<<< HEAD
as event-based cameras (ECs) [@Lichtsteiner2008] become increasingly capable and commercially available [@PoschEtAl_2014_Retinomorphic].
ECs build upon decades of research on retinomorphic vision, display superior
characteristics compared to frame-based cameras (FCs) and highlight the advantages of neuromorphic perception.
Here, we present `Pyrception` -- a Python library for bio-plausible retinomorphic modelling, simulation and processing
of raw visual input. The library builds upon a generalisation of sparse convolution using Toeplitz matrices
=======
as event-driven cameras (EDs) [@Lichtsteiner2008] become increasingly capable and commercially available [@PoschEtAl_2014_Retinomorphic].
EDs build upon decades of research on retinomorphic vision, display superior
characteristics in latency (microseconds) and power consumption (mWs to few Ws) compared to frame-based cameras (FCs) and highlight the advantages of neuromorphic perception.
Here, we present `Pyrception` -- a Python library for the retinal bio-plausible modelling, simulation and processing
of raw visual input, aimed at extending and enhancing bioinspired visual processing architectures. The library builds upon a generalisation of sparse convolution using Toeplitz matrices
>>>>>>> 3bc1badc9b0620e1d4db4cfc734b1c928a36973b
(outlined below), which allows for efficient processing and modelling of various features of the mammalian retina, such as
complex receptive fields, eccentricity-dependent receptive field sizes and log-polar cell arrangement [@chessa2016space].

# Statement of need

Hardware prototyping is a difficult and time-consuming process. This is especially valid for neuromorphic
vision sensors as the technology is new, and investing effort into developing a new sensor implementing innovative
on-sensor features may not be justified unless it has already been demonstrated that the processing pipeline works as intended.
`Pyrception` can be used to simulate retinal processing circuits, with envisioned applications in
the fields of neuromorphic vision and neuroscience.
The ability to perform accurate simulations of various retinal circuits is an efficient
way to demonstrate any potential advantages of such circuits and subsequently presenting a strong case for their
implementation in hardware. Similarly, such simulation can prove beneficial in neuroscience, for instance, for
modelling atypical or poorly understood visual mechanisms to gain insight into the causes and mechanisms behind of certain vision-related medical conditions.

# Implementation

The implemention of convolution in deep learning libraries such as [PyTorch](https://pytorch.org/) [@paszke2017automatic]
and [TensorFlow](https://www.tensorflow.org/) [@tensorflow2015-whitepaper] is based on the `Im2Col` mechanism [@ChellapillaEtAl_2006_HighPerformance].
Since in the case of digital visual input the convolution operation is discrete, in the `Im2Col` approach the input (an image) is partitioned into patches corresponding to each convolved location, which is determined by factors such as kernel size, stride, dilation and so forth. Each of these patches is unrolled into a column vector and concatenated to form the columns of a dense matrix, hence the term `Im2Col` (Fig. \ref{fig:convolution}a).
The same operation is performed on all kernels -- they are unrolled and concatenated as the _rows_ of a separate dense matrix (Fig. \ref{fig:convolution}b).
This operation reduces the convolution operation to a simple dense matrix-matrix multiplication.

While the `Im2Col` approach makes the convolution operation well suited for implementation on GPUs, it has several limitations
that make it unsuitable for modelling the working principle of the retina. For instance, `Im2Col` requires that all kernels be of the same size,
precluding the emulation of log-polar dependent receptive field architecture. Similarly, it makes it exceedingly difficult to implement non-rectangular
(e.g., elliptic or irregular) kernels.

In contrast, in `Pyrception` the image is not segmented into patches. Instead, it is unrolled into a single long vector by concatenating all
columns of the image (Fig. \ref{fig:convolution}c). As a consequence, the parts of the image that correspond to the receptive field of a kernel
are no longer contiguous. To be able to still perform convolution as a simple matrix-matrix (strictly, matrix-vector) multiplication,
each corresponding kernel must be unrolled into a \textit{sparse} row in such a way that multiplying the sparse row with the entire
image column vector gives the same result as if performed with the `Im2Col` approach (Fig. \ref{fig:convolution}d).
This type of convolution is based on Toeplitz matrices [@GnacikLapa_2022_UsingToeplitzMatrices],
where all diagonals contain identical elements, which is equivalent to shifting consecutive rows by one element to the right.
In `Pyrception`, kernels are unravelled into sparse rows, one for each convolved patch of the image, resulting in a sparse Toeplitz matrix.

<<<<<<< HEAD
The versatility of `Pyrception` stems from using a _generalised_ version of the sparse convolution approach with a Toeplitz matrix. The 'generalised' part refers to the fact that each patch of the image can be convolved with a _different_ kernel since each sparse rows in the kernel matrix encodes an entire kernel. Since each row in the kernel matrix has the same length as the image vector, each kernel can potentially have a full 'view' of the entire image. This gives us complete freedom to choose the parameters of each kernel independently, making it possible to construct convolutional matrices populated with arbitrary kernels with different sizes, orientations and arrangements. Notably, the kernels themselves can be sparse, making it possible to convolve sparse inputs (e.g., inputs arriving from an event camera).
=======
The versatility of `Pyrception` stems from \textit{sparse convolution} using a _generalised_ version of sparse convolution with a Toeplitz matrix. The 'generalised' part comes from the realisation that the kernel corresponding to each convolved patch can be _different_ since each patch in the image corresponds to a single sparse row in the kernel matrix. Even if the entire image is convolved with a single kernel (i.e., the same kernel applied to all locations of the image), the kernel must be unrolled and staggered multiple times into separate rows of the sparse kernel matrix to cover the entire image and obtain the same result as with the dense equivalents. The resulting sparse kernel matrix is much larger than the corresponding dense kernel matrix for equivalent dense convolution. However, this is compensated for by the complete freedom in choosing the kernel size, shape and orientation for each convolved patch of the image. Since each row in the kernel matrix has the same length as the image vector, each kernel can potentially have a full 'view' of the entire image. This gives us complete freedom in choosing what parts of the image each kernel convolves, the size and orientation of the kernel, and so forth. This makes it possible to construct convolutional matrices populated with a rich variety of kernels with different sizes, orientations and arrangements. Notably, the kernels themselves can be sparse, making it possible to convolve sparse inputs (e.g., inputs arriving from an event camera).
>>>>>>> 3bc1badc9b0620e1d4db4cfc734b1c928a36973b

Currently, `Pyrception` implements all five major layers of the mammalian retina, each of which can be emulated with different types of
cell arrangements and dynamics, as well as excitatory / inhibitory connections forming receptive fields of different shapes, sizes and orientations.
With the default parameters, the `Pyrception` layers implement a generic retinal processing pipeline consisting of a receptor layer followed by horizontal, bipolar,
amacrine and ganglion cell layers [@Masland_2011_CellPopulationsRetina]. By simply deriving from the existing layer classes and overriding the `forward()` method, it is possible to implement different retinal circuits, such as motion segmentation [@clerico2024retina;@olveczky2003segregation;@baccus2008retinal], looming detection for detecting approaching objects [@GollischEtAl_2010_EyeSmarter], optical flow estimation and visual tracking [@JavierTraverEtAl_2010_ReviewLogpolar;@angelo2025wandering]. We welcome contributions to `Pyrception` that would make the library as useful as possible, including implementing bio-plausible modalities other than vision.

# Figures

<<<<<<< HEAD
![Visualisation of dense and sparse operations on images and kernels for convolution. (a) Unrolling image patches into a dense matrix (`Im2Col` operation) and (b) a corresponding kernel into a dense row (`Ker2Row` operation). (c) Unrolling an image column-wise into a vector (`Im2Vec` operation). The (non-contiguous) elements corresponding to the same patches as those in (a) are illustrated. (d) Unrolling a kernel into a sparse row (`Ker2SpRow` operation). (e, f) Examples of circular kernels with radii of 3 and 5 pixels, respectively, with the corresponding unrolled `Ker2SpRow` representations shown below each kernel.\label{fig:convolution}](images/image-unrolling.png)
=======
![Visualisation of dense and sparse operations on images and kernels for convolution. (a) Unrolling image patches into a dense matrix (`Im2Col` operation) and (b) a corresponding kernel into a dense row (`Ker2Row` operation). (c) Unrolling an image column-wise into a vector (`Im2Vec` operation). The (non-contiguous) elements corresponding to the same patches as those in (a) are illustrated. (d) Unrolling a kernel into a sparse row (`Ker2SpRow` operation). (e, f) Examples of circular kernels with radii of 3 and 5 pixels, respectively, with the corresponding unrolled `Ker2SpRow` representations are shown below each kernel.\label{fig:convolution}](images/image-unrolling.png)
>>>>>>> 3bc1badc9b0620e1d4db4cfc734b1c928a36973b

![A visualisation of kernels of different sizes and orientations. (a) Kernels arranged and a phyllotactic log-polar distribution (similar to the seeds of a sunflower). Kernels towards the center of the image are smaller than ones towards the edges, emulating the eccentricity-dependent receptive field size observed in the mammalian retina. Due to the different sizes of the kernels, the 'stride' (and therefore the overlap) is non-uniform. (b) A zoomed-in version of the top left corner of the plot in (a). (c) A plot of the kernels for a subset of 10 sectors of kernels (a sector is a group of kernels whose centers are located along the same radial line). (d) The same kernel arrangement as in (a), but with elliptic kernels oriented at $45^{\circ}$. (e) A raw image (credit: [Von.grzanka / Wikimedia](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg)) and (f) the corresponding activation map resulting from convolving the image with the kernel map in (a). \label{fig:demo}](images/demo.png)

# Acknowledgements

A.H. is grateful for the support of the [Advanced Concepts Team of the European Space Agency](https://www.esa.int/gsp/ACT/), where the conceptualisation and a large portion of the development of this library took place. G.D. acknowledges the financial support from the European Union’s HORIZON-MSCA-2023-PF-01-01 research and innovation programme under the Marie Skłodowska-Curie grant agreement ENDEAVOR No 101149664.

# References
