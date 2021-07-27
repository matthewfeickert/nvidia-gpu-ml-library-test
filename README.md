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

* Lenovo ThinkPad X1 Extreme laptop
   - OS: Ubuntu 20.04
   - CPU: Intel Core i7-9750H @ 12x 4.5GHz
   - GPU: GeForce GTX 1650 with Max-Q Design
   - NVIDIA Driver: 470
   - Python: 3.9.6 built from source
* Custom built dekstop
   - OS: Ubuntu 20.04
   - CPU: AMD Ryzen 9 3900X 12-Core @ 24x 3.906GHz
   - GPU: GeForce RTX 2080 Ti
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
After the driver has finished installing, restart the computer to verify the driver has been installed successfully.
If you run

```
nvidia-smi
```

from the command line the displayed driver version should match the one you installed.

**N.B.:** To check all the GPUs that are currently visible to NVIDIA you can use

```
nvidia-smi --list-gpus
```

See the output of `nvidia-smi --help` for more details.

**Example:**

```
$ nvidia-smi --list-gpus
GPU 0: GeForce GTX 1650 with Max-Q Design (UUID: GPU-7061202f-798a-193c-6ff4-a6131eef00d3)
```

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
driver   : nvidia-driver-460 - distro non-free
driver   : nvidia-driver-460-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-470 - distro non-free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin
```

You can now either install the supported driver you want directly through `apt`

**Example:**

```
sudo apt-get install nvidia-driver-470
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

JAX expects to find the CUDA directory structure (what is will assign as the environment variable `CUDA_DIR`) under the path `/usr/local/cuda-x.x` where `x.x` is the CUDA version number that `nvcc --version` gives (`10.1` for example).
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

**WARNING:** This section will be out of date fast, so you'll have to adopt it for your particular circumstances.

TensorFlow requires the [NVIDIA cuDNN](https://developer.nvidia.com/CUDNN) closed source libraries, which are a pain to get and have quite bad documentation.
To download the libraries you will need to make an account with NVIDIA and register as a developer, which is also a bad experience.
Once you've done that go to the [cuDNN download page](https://developer.nvidia.com/rdp/cudnn-download), agree to the Software License Agreement, and the select the version of cuDNN that matches the version of CUDA your **operating system** has (**the version from `nvidia-smi`** which is not necessarily the same as the version from `nvcc --version`)

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

#### Adding CUDA and cuDNN to PATHs

The installed libraries should also be known added to `PATH` and `LD_LIBARY_PATH`, so add the following to your `~/.profile` to be loaded at system login

```
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
For example, looking at supported ranges for TensorFlow `v2.3.0` and `v2.4.0`

| Version          | Python version | Compiler  | Build tools | cuDNN | CUDA |
|------------------|----------------|-----------|-------------|-------|------|
| tensorflow-2.4.0 | 3.6-3.8        | GCC 7.3.1 | Bazel 3.1.0 | 8.0   | 11.0 |
| tensorflow-2.3.0 | 3.5-3.8        | GCC 7.3.1 | Bazel 3.1.0 | 7.6   | 10.1 |

it is seen that for our example of

```
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

```
sudo ln -s /usr/lib/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.7
```

You should now have a directory structure for `usr/local/cuda` that looks something like the following

```
$ tree /usr/local/cuda
/usr/local/cuda
├── bin
│   └── nvcc -> /usr/bin/nvcc
├── include
│   ├── cuda.h -> /usr/include/cuda.h
│   ├── cudnn_adv_infer.h
│   ├── cudnn_adv_train.h
│   ├── cudnn_backend.h
│   ├── cudnn_cnn_infer.h
│   ├── cudnn_cnn_train.h
│   ├── cudnn.h
│   ├── cudnn_ops_infer.h
│   ├── cudnn_ops_train.h
│   └── cudnn_version.h
├── lib64
│   ├── libcudnn_adv_infer.so
│   ├── libcudnn_adv_infer.so.8
│   ├── libcudnn_adv_infer.so.8.0.5
│   ├── libcudnn_adv_train.so
│   ├── libcudnn_adv_train.so.8
│   ├── libcudnn_adv_train.so.8.0.5
│   ├── libcudnn_cnn_infer.so
│   ├── libcudnn_cnn_infer.so.8
│   ├── libcudnn_cnn_infer.so.8.0.5
│   ├── libcudnn_cnn_train.so
│   ├── libcudnn_cnn_train.so.8
│   ├── libcudnn_cnn_train.so.8.0.5
│   ├── libcudnn_ops_infer.so
│   ├── libcudnn_ops_infer.so.8
│   ├── libcudnn_ops_infer.so.8.0.5
│   ├── libcudnn_ops_train.so
│   ├── libcudnn_ops_train.so.8
│   ├── libcudnn_ops_train.so.8.0.5
│   ├── libcudnn.so
│   ├── libcudnn.so.7 -> /usr/lib/cuda/lib64/libcudnn.so.8
│   ├── libcudnn.so.8
│   ├── libcudnn.so.8.0.5
│   └── libcudnn_static.a
├── nvvm
│   └── libdevice -> ../../nvidia-cuda-toolkit/libdevice
└── version.txt

5 directories, 35 files
```

With this final set of libraries installed restart your computer.

## Testing

### Detect GPU

For all of the ML libaries you can now run the `x_detect_GPU.py` tests which test that the library can properly access the GPU and CUDA, where `x` is the library name/nickname.

### MNIST

For all of the ML libraries you can run a simple MNIST test by running `x_MNIST.py`, where `x` is the library name/nickname.

### Monitoring

It is worthwhile in another terminal to watch the GPU performance with `nvidia-smi` while running tests

```
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
