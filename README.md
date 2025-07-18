# KMeans Computation on Images using NVIDIA CUFFT Signal Processing with CUDA

## Overview

This project demonstrates the use of the cuFFT (CUDA Fast Fourier Transform) library to perform signal processing on 2D image data using GPU acceleration. The primary objective is to apply forward and inverse FFTs to analyze and manipulate frequency-domain representations of images efficiently on modern NVIDIA GPUs.

As part of the CUDA at Scale for the Enterprise course, this project serves as a practical template for learning how to implement and optimize basic frequency-domain operations using CUDA and cuFFT. The techniques demonstrated are fundamental in many real-world applications such as image filtering, compression, denoising, and feature extraction.

## Code Organization

```bin/```
./cuda_K-means.o object file executable

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```images/```
Input images .pgm dataset containing 10 large images on which operation has been performed upon


```lib/```
All .h files are placed here 

```src/```
src/common.h
src/cuda_K-means.cu (main file)
src/cuda_runtime.h (for spare purpose)
src/imageRotationNPP.cpp though not used in this project but can be checked out for NPP use cases.
src/stb_image_write.h
src/stb_image.h
src/timer.h 

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
1. sudo apt install nvidia-cuda-toolkit
2. sudo apt install nvidia-smi
3. sudo apt install nvidia-driver

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

## Key Concepts

Performance Strategies, Image Processing, Image segmentation and Filteration, cufft Library, Fast-Fourier-Transform with Inverse Fast-Fourier-Transform.

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved
cuda_runtime.h and stb_image.h 

## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run
just build by make and do a ./cuda_K-means

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```


## Running the Program
After building the project, you can run the program using the following command:

```bash
Copy code
make run
```

This command will execute the compiled binary, rotating the input image (Lena.png) by 45 degrees, and save the result as Lena_rotated.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
- Copy code
./bin/imageRotationNPP --input data/Lena.png --output data/Lena_rotated.png
```
(but Remember the above instruction is for NPP rotation version)

- Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
- Copy code
make clean
```

This will remove all files in the bin/ directory.
