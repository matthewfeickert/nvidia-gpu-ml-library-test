from torch import cuda

if __name__ == "__main__":
    # c.f. https://chrisalbon.com/deep_learning/pytorch/basics/check_if_pytorch_is_using_gpu/
    print(f"Number of GPUs found on system: {cuda.device_count()}")
    is_gpu_available = cuda.is_available()
    print(f"PyTorch has active GPU: {is_gpu_available}")
    if is_gpu_available:
        print(f"Active GPU index: {cuda.current_device()}")
        print(f"Active GPU name: {cuda.get_device_name(cuda.current_device())}")
