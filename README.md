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

JAX expectes to find the CUDA directory structure (what is will assign as the environment variable `CUDA_DIR`) under the path `/usr/local/cuda-x.x` where `x.x` is the CUDA version number that `nvcc --version` gives (`10.1` for example).
If NVIDIA CUDA Toolkit was installed through the `nvidia-cuda-toolkit` Ubuntu package CUDA will instead be found under `/usr/lib/cuda`.
To make CUDA findable to JAX a symlink can be created

```
sudo ln -s /usr/lib/cuda /usr/local/cuda-x.x
```

**Example:**

```
sudo ln -s /usr/lib/cuda /usr/local/cuda-10.1
```

To test the location of the installed CUDA release you can set the following environment variable before importing JAX

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda
```

**Example:**

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/ python jax_MNIST.py
```

If you have questions on this step refer to the [relevant section in the JAX README](https://github.com/google/jax#pip-installation).

#### TensorFlow

**WARNING:** This section will be out of date fast, so you'll have to adopt it for your particular circumastnaces.

TensorFlow requires the [NVIDIA cuDNN](https://developer.nvidia.com/CUDNN) closed source libraries, which are a pain to get and have quite bad documentation.
To download the libraries you will need to make an account with NVIDIA and register as a developer, which is also a bad experience.
Once you've done that go to the [cuDNN download page](https://developer.nvidia.com/rdp/cudnn-download), agreee to the Software License Agreement, and the select the version of cuDNN that matches the version of CUDA your **operating system** has (**the version from `nvidia-smi`** which is not necesarilly the same as the version from `nvcc --version`)

**Example:**

For the choices of

- cuDNN v8.0.5 for CUDA 11.1
- cuDNN v8.0.5 for CUDA 11.0
- cuDNN v8.0.5 for CUDA 10.2
- cuDNN v8.0.5 for CUDA 10.1

```
$ nvidia-smi | grep "CUDA Version"
| NVIDIA-SMI 455.45.01    Driver Version: 455.45.01    CUDA Version: 11.1
```

would indicate that cuDNN v8.0.5 for CUDA 11.1 is the recommended version.
(This is verified by noting that when clicked on the entry for cuDNN v8.0.5 for CUDA 11.1 lists support for Ubuntu 20.04, but the entry for cuDNN v8.0.5 for CUDA 10.1 lists support only for Ubuntu 18.04.)

Click on the cuDNN release you want to download to see the available libraries for supportes sytem architectures.
As these instructions are using Ubuntu, download the tarballs and Debian binaries for cuDNN libary and the cuDNN runtime library, developer library, and code samples.

**Example:**

- cuDNN Library for Linux (x86_64)
- cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Code Samples and User Guide for Ubuntu20.04 x86_64 (Deb)

Once all the libraries are downloaded locally refer to the [directions for installing on Linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux) in the [cuDNN installation guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).
The documentation refers to a CUDA directory path (which they generically call `/usr/local/cuda`) and a download path for all of the cuDNN libraries (referred to as `<cudnnpath>`).
For the CUDA directory path we _could_ use our existing symlink of `/usr/local/cuda-10.1`, but the cuDNN examples all assume the path is `/usr/local/cuda` so it is eaiser to make a new symlink of `/usr/local/cuda` pointing to `/usr/lib/cuda`.

```
sudo ln -s /usr/lib/cuda /usr/local/cuda
```

The examples are also going to assume that `nvcc` is at `/usr/local/cuda/bin/nvcc` and `cuda.h` is at `/usr/local/cuda/include/cuda.h`, so make additional symlinks of those paths pointing to `/usr/bin/nvcc` and `/usr/include/cuda.h`

```
sudo ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
sudo ln -s /usr/include/cuda.h /usr/local/cuda/include/cuda.h
```

#### Install cuDNN Library

1. Navigate to your `<cudnnpath>` directory containing the cuDNN tar file (exmple: `cudnn-11.1-linux-x64-v8.0.5.39.tgz`)
2. Untar the cuDNN libary tarball (the untarred directory name is `cuda`)

```
tar -xzvf cudnn-*-linux-x64-v*.tgz
```

3. Copy the library files into the CUDA Toolkit directory

```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

4. Set the permissions for the files to be universally readable

```
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### Install cuDNN Runtime and Developer Libraries

To use in your applications the cuDNN runtime library, developer library, and code samples should be installed too.
This can be done with `apt install` from your `<cudnnpath>`.

**Example:**

```
sudo apt install ./libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb
sudo apt install ./libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb
sudo apt install ./libcudnn8-samples_8.0.5.39-1+cuda11.1_amd64.deb
```

#### Test cuDNN Installation

Copy the cuDNN samples to a writable path

```
cp -r /usr/src/cudnn_samples_v8/ $PWD
```

then navigate to the `mnistCUDNN` sample directory and compile and run the sample

```
cd cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```

If everything is setup correctly then the resulting output should conclude with

```
Test passed!
```

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
