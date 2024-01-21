# NVIDIA GPU ML library test

Simple tests for JAX, PyTorch, and TensorFlow to test if the installed NVIDIA drivers are being properly picked up.

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/matthewfeickert/nvidia-gpu-ml-library-test/main.svg)](https://results.pre-commit.ci/latest/github/matthewfeickert/nvidia-gpu-ml-library-test/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Requirements

These instructions assume working on [Ubuntu 20.04 LTS](https://releases.ubuntu.com/20.04/).

- Computer with NVIDIA GPU installed.
- Linux operating system (assumed to be an Ubuntu LTS) with root access.
- Python 3.6+ installed  (recommended through [pyenv](https://github.com/pyenv/pyenv) for easy configuration).

**Example:**

This setup has been tested on the following systems:

* Dell XPS 15 9510 laptop
   - OS: Ubuntu 22.04
   - CPU: 11th Gen Intel Core i9-11900H @ 16x 4.8GHz
   - GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
   - NVIDIA Driver: 535
   - Python: 3.10.6 built from source
* Custom built desktop
   - OS: Ubuntu 20.04
   - CPU: AMD Ryzen 9 3900X 12-Core @ 24x 3.906GHz
   - GPU: GeForce RTX 2080 Ti
   - NVIDIA Driver: 455
   - Python: 3.8.6 built from source

## Setup

### Installing Python Libraries

Create a Python virtual environment and install the base libraries from the relevant `requirements.txt` files.

Examples:

* To install the relevant JAX libraries for use with NVIDIA GPUs

```console
python -m pip install -r requirements-jax.txt
```

* To install the relevant JAX libraries for use with Apple silicon GPUs

```console
python -m pip install -r requirements-jax-metal.txt
```

### Installing NVIDIA Drivers and CUDA Libraries

#### Ubuntu NVIDIA Drivers

##### Ubuntu's Software & Updates Utility

The easiest way to determine the correct NVIDIA driver for your system is to have it determine it automatically through Ubuntu's [Software & Updates utility and selecting the Drivers tab](https://wiki.ubuntu.com/SoftwareAndUpdatesSettings).

> The "Drivers" tab should begin with a listbox containing a progress bar and the text "Searching for available drivers…" until the search is complete.
> Once the search is complete, the listbox should list each device for which proprietary drivers could be installed.
> Each item in the list should have an indicator light: green if a driver tested with that Ubuntu release is being used, yellow if any other driver is being used, or red if no driver is being used.

Select the recommended NVIDIA driver from the list (proprietary, tested) and then select "Apply Changes" to install the driver.
After the driver has finished installing, restart the computer to verify the driver has been installed successfully.
If you run

```console
nvidia-smi
```

from the command line the displayed driver version should match the one you installed.

**N.B.:** To check all the GPUs that are currently visible to NVIDIA you can use

```console
nvidia-smi --list-gpus
```

See the output of `nvidia-smi --help` for more details.

**Example:**

```console
$ nvidia-smi --list-gpus
GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU (UUID: GPU-9b3a1382-1fb8-43c7-67b1-c28af22b6767)
```

##### Command Line

Alternatively, if you are running headless or over a remote connection you can determine and install the correct driver from the command line.
From the command line run

```console
ubuntu-drivers devices
```

to get a list of all devices on the machine that need drivers and the recommended drivers.

**Example:**

```console
$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd000025A0sv00001028sd00000A61bc03sc02i00
vendor   : NVIDIA Corporation
model    : GA107M [GeForce RTX 3050 Ti Mobile]
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-525-open - distro non-free
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-535-open - distro non-free
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-525-server - distro non-free
driver   : nvidia-driver-525 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

You can now either install the supported driver you want directly through `apt`

**Example:**

```console
sudo apt-get install nvidia-driver-535
```

or you can let `ubnutu-driver` install the recommended driver for you automatically

```console
sudo ubuntu-drivers autoinstall
```

#### NVIDIA CUDA Toolkit

After installing the NVIDIA driver, the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) also needs to be installed.
This needs to be done every time you update the NVIDIA driver.
This can be done manually by following the instructions on the NVIDIA website, but it can also be done automatically through `apt` installing the [Ubuntu package `nvidia-cuda-toolkit`](https://packages.ubuntu.com/search?keywords=nvidia-cuda-toolkit).

```console
sudo apt-get update -y
sudo apt-get install -y nvidia-cuda-toolkit
```

**Example:**

```console
$ apt show nvidia-cuda-toolkit | head -n 5
Package: nvidia-cuda-toolkit
Version: 11.5.1-1ubuntu1
Priority: extra
Section: multiverse/devel
Origin: Ubuntu
```

After the NVIDIA CUDA Toolkit is installed restart the computer.

**N.B.:** If the NVIDIA drivers are ever changed the NVIDIA CUDA Toolkit will need to be reinstalled.


Now that the system NVIDIA drivers are installed the necessary requirements can be stepped through or the different machine learning backends in order (from easiest to hardest).

#### PyTorch

PyTorch makes things very easy by [packaging all of the necessary CUDA libraries with its binary distributions](https://discuss.pytorch.org/t/newbie-question-what-are-the-prerequisites-for-running-pytorch-with-gpu/698/3) (which is why they are so huge).
So by `pip` installing the `torch` wheel all necessary libraries are installed.


#### JAX

The [CUDA and CUDNN release wheels can be installed from PyPI and Google with `pip`](https://github.com/google/jax/blob/6eb3096461abdbf622df5ebeee57ee40bdfb66b0/README.md#pip-installation-gpu-cuda-installed-via-pip-easier)

```console
python -m pip install --upgrade "jax[cuda12_pip]" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

##### With local CUDA installations

To instead install the `jax` and `jaxlib` but use locally installed CUDA and CUDNN versions follow the instructions in the [JAX README](https://github.com/google/jax/blob/main/README.md).
In these circumstances to test the location of the installed CUDA release you can set the following environment variable before importing JAX

```console
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda
```

**Example:**

```console
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/ python jax_MNIST.py
```

#### TensorFlow

**WARNING:** This section will be out of date fast, so you'll have to adopt it for your particular circumstances.

TensorFlow requires the [NVIDIA cuDNN](https://developer.nvidia.com/CUDNN) closed source libraries, which are a pain to get and have quite bad documentation.
To download the libraries you will need to make an account with NVIDIA and register as a developer, which is also a bad experience.
Once you've done that go to the [cuDNN download page](https://developer.nvidia.com/rdp/cudnn-download), agree to the Software License Agreement, and the select the version of cuDNN that matches the version of CUDA your **operating system** has (**the version from `nvidia-smi`** which is not necessarily the same as the version from `nvcc --version`)

**Example:**

For the choices of

- cuDNN v8.2.2 for CUDA 11.4
- cuDNN v8.2.2 for CUDA 10.2

```console
$ nvidia-smi | grep "CUDA Version"
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
```

would indicate that cuDNN v8.2.2 for CUDA 11.4 is the recommended version.
(This is verified by noting that when clicked on the entry for cuDNN v8.2.2 for CUDA 11.4 lists support for Ubuntu 20.04, but the entry for cuDNN v8.2.2 for CUDA 10.2 lists support only for Ubuntu 18.04.)

Click on the cuDNN release you want to download to see the available libraries for supports system architectures.
As these instructions are using Ubuntu, download the tarballs and Debian binaries for cuDNN library and the cuDNN runtime library, developer library, and code samples.

**Example:**

- cuDNN Library for Linux (x86_64)
- cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)
- cuDNN Code Samples and User Guide for Ubuntu20.04 x86_64 (Deb)

Once all the libraries are downloaded locally refer to the [directions for installing on Linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux) in the [cuDNN installation guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).
The documentation refers to a CUDA directory path (which they generically call `/usr/local/cuda`) and a download path for all of the cuDNN libraries (referred to as `<cudnnpath>`).
For the CUDA directory path we _could_ use our existing symlink of `/usr/local/cuda-10.1`, but the cuDNN examples all assume the path is `/usr/local/cuda` so it is easier to make a new symlink of `/usr/local/cuda` pointing to `/usr/lib/cuda`.

```console
sudo ln -s /usr/lib/cuda /usr/local/cuda
```

The examples are also going to assume that `nvcc` is at `/usr/local/cuda/bin/nvcc` and `cuda.h` is at `/usr/local/cuda/include/cuda.h`, so make additional symlinks of those paths pointing to `/usr/bin/nvcc` and `/usr/include/cuda.h`

```console
sudo ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
sudo ln -s /usr/include/cuda.h /usr/local/cuda/include/cuda.h
```

#### Install cuDNN Library

1. Navigate to your `<cudnnpath>` directory containing the cuDNN tar file (example: `cudnn-11.4-linux-x64-v8.2.2.26.tgz`)
2. Untar the cuDNN library tarball (the untarred directory name is `cuda`)

```console
tar -xzvf cudnn-*-linux-x64-v*.tgz
```

3. Copy the library files into the CUDA Toolkit directory

```console
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

4. Set the permissions for the files to be universally readable

```console
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### Install cuDNN Runtime and Developer Libraries

To use in your applications the cuDNN runtime library, developer library, and code samples should be installed too.
This can be done with `apt install` from your `<cudnnpath>`.

**Example:**

```console
sudo apt install ./libcudnn8_8.2.2.26-1+cuda11.4_amd64.deb
sudo apt install ./libcudnn8-dev_8.2.2.26-1+cuda11.4_amd64.deb
sudo apt install ./libcudnn8-samples_8.2.2.26-1+cuda11.4_amd64.deb
```

#### Test cuDNN Installation

Copy the cuDNN samples to a writable path

```console
cp -r /usr/src/cudnn_samples_v8/ $PWD
```

then navigate to the `mnistCUDNN` sample directory and compile and run the sample

```console
cd cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```

If everything is setup correctly then the resulting output should conclude with

```
Test passed!
```

#### Adding CUDA and cuDNN to PATHs

The installed libraries should also be known added to `PATH` and `LD_LIBARY_PATH`, so add the following to your `~/.profile` to be loaded at system login

```bash
# Add CUDA Toolkit 10.1 to PATH
# /usr/local/cuda-10.1 should be a symlink to /usr/lib/cuda
if [ -d "/usr/local/cuda-10.1/bin" ]; then
    PATH="/usr/local/cuda-10.1/bin:${PATH}"; export PATH;
elif [ -d "/usr/lib/cuda/bin" ]; then
    PATH="/usr/lib/cuda/bin:${PATH}"; export PATH;
fi
# Add cuDNN to LD_LIBRARY_PATH
# /usr/local/cuda should be a symlink to /usr/lib/cuda
if [ -d "/usr/local/cuda/lib64" ]; then
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH;
elif [ -d "/usr/lib/cuda/lib64" ]; then
    LD_LIBRARY_PATH="/usr/lib/cuda/lib64:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH;
fi
```

#### Check TensorFlow Version Restrictions

TensorFlow does not really respect semvar as minor releases act essentially as major releases with breaking changes.
This comes into play when considering the [tested build configurations for CUDA and cuDNN versions](https://www.tensorflow.org/install/source#linux).
For example, looking at supported ranges for TensorFlow `v2.3.0` through `v2.5.0`

| Version          | Python version | Compiler  | Build tools | cuDNN | CUDA |
|------------------|----------------|-----------|-------------|-------|------|
| tensorflow-2.5.0 | 3.6-3.9        | GCC 7.3.1 | Bazel 3.7.2 | 8.1   | 11.2 |
| tensorflow-2.4.0 | 3.6-3.8        | GCC 7.3.1 | Bazel 3.1.0 | 8.0   | 11.0 |
| tensorflow-2.3.0 | 3.5-3.8        | GCC 7.3.1 | Bazel 3.1.0 | 7.6   | 10.1 |

it is seen that for our example of

```console
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

only TensorFlow `v2.3.0` will be compatible with out installation.
However, TensorFlow `v2.3.0` requires cuDNN `v7.X` (`libcudnn.so.7`) and we have cuDNN `v8.x` (`libcudnn.so.8`).
The NVIDIA [cuDNN installation documentation notes](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#upgrade) that

> Since version 8 can coexist with previous versions of cuDNN, if the user has an older version of cuDNN such as v6 or v7, installing version 8 will not automatically delete an older revision.

While we could go and try to install cuDNN `v7.6` from the [cuDNN archives](https://developer.nvidia.com/rdp/cudnn-archive) it turns out that [TensorFlow is okay with](https://github.com/tensorflow/tensorflow/issues/20271#issuecomment-643296453) symlinking `libcudnn.so.8` to a target of `libcudnn.so.7`, so until this causes problems move forward with this approach

```console
sudo ln -s /usr/lib/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.7
```

You should now have a directory structure for `usr/local/cuda` that looks something like the following

```console
$ tree /usr/local/cuda
/usr/local/cuda
├── bin
│   └── nvcc -> /usr/bin/nvcc
├── include
│   ├── cuda.h -> /usr/include/cuda.h
│   ├── cudnn_adv_infer.h
│   ├── cudnn_adv_infer_v8.h
│   ├── cudnn_adv_train.h
│   ├── cudnn_adv_train_v8.h
│   ├── cudnn_backend.h
│   ├── cudnn_backend_v8.h
│   ├── cudnn_cnn_infer.h
│   ├── cudnn_cnn_infer_v8.h
│   ├── cudnn_cnn_train.h
│   ├── cudnn_cnn_train_v8.h
│   ├── cudnn.h
│   ├── cudnn_ops_infer.h
│   ├── cudnn_ops_infer_v8.h
│   ├── cudnn_ops_train.h
│   ├── cudnn_ops_train_v8.h
│   ├── cudnn_v8.h
│   ├── cudnn_version.h
│   └── cudnn_version_v8.h
├── lib64
│   ├── libcudnn_adv_infer.so
│   ├── libcudnn_adv_infer.so.8
│   ├── libcudnn_adv_infer.so.8.2.2
│   ├── libcudnn_adv_train.so
│   ├── libcudnn_adv_train.so.8
│   ├── libcudnn_adv_train.so.8.2.2
│   ├── libcudnn_cnn_infer.so
│   ├── libcudnn_cnn_infer.so.8
│   ├── libcudnn_cnn_infer.so.8.2.2
│   ├── libcudnn_cnn_infer_static.a
│   ├── libcudnn_cnn_infer_static_v8.a
│   ├── libcudnn_cnn_train.so
│   ├── libcudnn_cnn_train.so.8
│   ├── libcudnn_cnn_train.so.8.2.2
│   ├── libcudnn_cnn_train_static.a
│   ├── libcudnn_cnn_train_static_v8.a
│   ├── libcudnn_ops_infer.so
│   ├── libcudnn_ops_infer.so.8
│   ├── libcudnn_ops_infer.so.8.2.2
│   ├── libcudnn_ops_train.so
│   ├── libcudnn_ops_train.so.8
│   ├── libcudnn_ops_train.so.8.2.2
│   ├── libcudnn.so
│   ├── libcudnn.so.7 -> /usr/lib/cuda/lib64/libcudnn.so.8
│   ├── libcudnn.so.8
│   ├── libcudnn.so.8.2.2
│   ├── libcudnn_static.a
│   └── libcudnn_static_v8.a
├── nvvm
│   └── libdevice -> ../../nvidia-cuda-toolkit/libdevice
└── version.txt

5 directories, 49 files
```

With this final set of libraries installed restart your computer.

## Testing

### Detect GPU

For all of the ML libraries you can now run the `x_detect_GPU.py` tests which test that the library can properly access the GPU and CUDA, where `x` is the library name/nickname.

### MNIST

For all of the ML libraries you can run a simple MNIST test by running `x_MNIST.py`, where `x` is the library name/nickname.

### Monitoring

It is worthwhile in another terminal to watch the GPU performance with `nvidia-smi` while running tests

```console
watch --interval 0.5 nvidia-smi
```

## Notes

### Useful Sites

- The [JAX README](https://github.com/google/jax)
- [TensorFlow GPU support page](https://www.tensorflow.org/install/gpu) which leads to the **actually useful** listing of [tested build configurations for CUDA and cuDNN versions](https://www.tensorflow.org/install/source#linux)

### Useful GitHub Issues

- [JAX Issue 3984](https://github.com/google/jax/issues/3984): automatic detection for GPU pip install doesn't quite work on ubuntu 20.04
- [TensorFlow Issue 20271](https://github.com/tensorflow/tensorflow/issues/20271#issuecomment-643296453): ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory

## Acknowledgements

Thanks to [Giordon Stark](https://github.com/kratsg/) who greatly helped me scaffold the right approach to this setup, as well as for his help doing system setup comparisons.
