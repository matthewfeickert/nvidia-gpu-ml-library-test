# Nvidia GPU ML library test

Simple tests for JAX, PyTorch, and TensorFlow to test if the installed Nvidia drivers are being properly picked up

## Setup

After installing the proper Nvidia drivers for your system create a Python virtual environment and install the base libraires

```
python -m pip install -r requirements.txt
```

From here the [GPU version of `jaxlib` will need to be installed](https://github.com/google/jax#pip-installation), and can be determined from the version of `jaxlib` that was installed from PyPI and the version of CUDA installed.
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
