# NVIDIA GPU ML library test

Simple tests for JAX, PyTorch, and TensorFlow to test if the installed NVIDIA drivers are being properly picked up.

## Setup

### Installing Python Libraries

Create a Python virtual environment and install the base libraries

```
python -m pip install -r requirements.txt
```

### Installing NVIDIA Drivers and CUDA Libraries

These instructions assume working on Ubuntu 20.04 LTS.

### Ubuntu NVIDIA Drivers

Now that the system NVIDIA drivers are installed the necessary requirements can be stepped through or the different machine learning backends in order (from easiest to hardest).

### PyTorch

PyTorch makes things very easy by [packaging all of the necessary CUDA libraries with its binary distirbutions](https://discuss.pytorch.org/t/newbie-question-what-are-the-prerequisites-for-running-pytorch-with-gpu/698/3) (which is why they are so huge).
So by `pip` installing the `torch` wheel all necessary libraries are installed.


### JAX

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

## Testing

It is worthwhile in another terminal watching the GPU performance with `nvidia-smi` while running tests

```
watch --interval 1 nvidia-smi
```

### Temporary Note

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

## Acknowledgements

Thanks to Giordon Stark who greatly helped me scafold the right approach to this setup, as well as for his help doing system setup comparisons.
