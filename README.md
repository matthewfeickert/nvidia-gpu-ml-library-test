# NVIDIA GPU ML library test

Simple tests for JAX, PyTorch, and TensorFlow to test if the installed NVIDIA drivers are being properly picked up.

## Requirements

These instructions assume working on [Ubuntu 20.04 LTS](https://releases.ubuntu.com/20.04/).

- Computer with NVIDIA GPU installed.
- Linux operating system (assumed to be an Ubuntu LTS) with root access.
- Python 3.6+ installed  (recommended through [pyenv](https://github.com/pyenv/pyenv) for easy configuration).

**Example:**

This setup has been tested on the following systems:

* Lenovo ThinkPad X1 Extreme laptop
   - OS: Ubuntu 20.04
   - GPU: GeForce GTX 1650 with Max-Q Design
   - NVIDIA Driver: 455
   - Python: 3.8.6 built from source

## Setup

### Installing Python Libraries

Create a Python virtual environment and install the base libraries

```
python -m pip install -r requirements.txt
```

### Installing NVIDIA Drivers and CUDA Libraries

#### Ubuntu NVIDIA Drivers

##### Ubuntu's Software & Updates Utility

The easiest way to determine the correct NVIDIA driver for your system is to have it determine it automatically through Ubuntu's [Software & Updates utility and selecting the Drivers tab](https://wiki.ubuntu.com/SoftwareAndUpdatesSettings).

> The "Drivers" tab should begin with a listbox containing a progress bar and the text "Searching for available drivers…" until the search is complete.
> Once the search is complete, the listbox should list each device for which proprietary drivers could be installed.
> Each item in the list should have an indicator light: green if a driver tested with that Ubuntu release is being used, yellow if any other driver is being used, or red if no driver is being used.

Select the recommended NVIDIA driver from the list (proprietary, tested) and then select "Apply Changes" to install the driver.
After the driver has finished installing, restart the computer to verify the driver has been installed succesfully.
If you run

```
nvidia-smi
```

from the command line the displayed driver version should match the one you installed.

##### Command Line

Alternatively, if you are running headless or over a remote connection you can determine and install the correct driver from the command line.
From the command line run

```
ubuntu-drivers devices
```

to get a list of all devices on the machine that need drivers and the recommended drivers.

**Example:**

```
$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001F91sv000017AAsd0000229Fbc03sc00i00
vendor   : NVIDIA Corporation
model    : TU117M [GeForce GTX 1650 Mobile / Max-Q]
driver   : nvidia-driver-450 - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-455 - distro non-free recommended
driver   : nvidia-driver-440-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

You can now either install the supported driver you want directly through `apt`

**Example:**

```
sudo apt-get install nvidia-driver-455
```

or you can let `ubnutu-driver` install the recommended driver for you automatically

```
sudo ubuntu-drivers autoinstall
```

#### NVIDIA CUDA Toolkit

After installing the NVIDIA driver, the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) also needs to be installed.
This can be done manually by following the instructions on the NVIDIA website, but it can also be done automatically through `apt` installing the [Ubuntu package `nvidia-cuda-toolkit`](https://packages.ubuntu.com/search?keywords=nvidia-cuda-toolkit).

```
sudo apt-get update -y
sudo apt-get install -y nvidia-cuda-toolkit
```

**Example:**

```
$ apt show nvidia-cuda-toolkit | head -n 5
Package: nvidia-cuda-toolkit
Version: 10.1.243-3
Priority: extra
Section: multiverse/devel
Origin: Ubuntu
```

After the NVIDIA CUDA Toolkit is installed restart the computer.

**N.B.:** If the NVIDIA drivers are ever changed the NVIDIA CUDA Toolkit will need to be reinstalled.


Now that the system NVIDIA drivers are installed the necessary requirements can be stepped through or the different machine learning backends in order (from easiest to hardest).

#### PyTorch

PyTorch makes things very easy by [packaging all of the necessary CUDA libraries with its binary distirbutions](https://discuss.pytorch.org/t/newbie-question-what-are-the-prerequisites-for-running-pytorch-with-gpu/698/3) (which is why they are so huge).
So by `pip` installing the `torch` wheel all necessary libraries are installed.


#### JAX

The [GPU version of `jaxlib` will need to be installed](https://github.com/google/jax#pip-installation), and can be determined from the version of `jaxlib` that was installed from PyPI and the version of CUDA installed.
The GPU release can be installed from Google with

```
python -m pip install --upgrade jax jaxlib==<jaxlib VERSION GOES HERE>+cuda<CUDA VERSION GOES HERE> --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
```

The version of `jaxlib` installed can be found from `pip` and the version of CUDA installed can be found from

```
nvcc --version
```

**Example:**

```
$ pip list | grep jax
jax                    0.2.7
jaxlib                 0.1.57
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

indicates that

```
python -m pip install --upgrade jax jaxlib==0.1.57+cuda101 --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
```

is needed.


### Temporary Note

**JAX**

If Nvidia CUDA Toolkit is installed the Ubuntu PPAs with

```
apt-get install nvidia-cuda-toolkit
```

CUDA can be found under `/usr/lib/cuda/`.
To deal with this for the time being run JAX commands prefecaed with

```
 XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/
```

so for example

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/ python jax_MNIST.py
```

Or better symlink to `/usr/local/cuda`

```
sudo ln -s /usr/lib/cuda /usr/local/cuda
```

#### TensorFlow

https://developer.nvidia.com/rdp/cudnn-download

Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1

Get all these

- cuDNN Library for Linux (x86_64)
- cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Code Samples and User Guide for Ubuntu20.04 x86_64 (Deb)

install .debs and then c.f. installation guide http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html which mentions 2.3.1. Tar File Installation

Procedure
> 1. Navigate to your <cudnnpath> directory containing the cuDNN tar file.
> 2. Unzip the cuDNN package.
> ```
> $ tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
> ```
> or
>
> ```
> $ tar -xzvf cudnn-x.x-linux-aarch64sbsa-v8.x.x.x.tgz
> ```
> 3 . Copy the following files into the CUDA Toolkit directory.
> ```
> $ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
> $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
> ```
>
> $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

To get libcudnn.so.7 whenhave .8 just

symlink libcudnn.so.8 to libcudnn.so.7
```
sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.7
```

## Testing

It is worthwhile in another terminal watching the GPU performance with `nvidia-smi` while running tests

```
watch --interval 1 nvidia-smi
```

## Notes

### Useful GitHub Issues

- [JAX Issue 3984](https://github.com/google/jax/issues/3984): automatic detection for GPU pip install doesn't quite work on ubuntu 20.04
- [TensorFlow Issue 20271](https://github.com/tensorflow/tensorflow/issues/20271#issuecomment-647113141): ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory

## Acknowledgements

Thanks to [Giordon Stark](https://github.com/kratsg/) who greatly helped me scafold the right approach to this setup, as well as for his help doing system setup comparisons.
