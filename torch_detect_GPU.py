import torch
from torch import cuda

if __name__ == "__main__":
    if torch.backends.cuda.is_built():
        # c.f. https://chrisalbon.com/deep_learning/pytorch/basics/check_if_pytorch_is_using_gpu/
        print(f"PyTorch build CUDA version: {torch.version.cuda}")
        print(f"PyTorch build cuDNN version: {torch.backends.cudnn.version()}")
        print(f"PyTorch build NCCL version: {torch.cuda.nccl.version()}")

        print(f"\nNumber of GPUs found on system: {cuda.device_count()}")

    if cuda.is_available():
        print(f"\nActive GPU index: {cuda.current_device()}")
        print(f"Active GPU name: {cuda.get_device_name(cuda.current_device())}")
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(f"PyTorch has active GPU: {mps_device}")
    else:
        print(f"PyTorch has no active GPU")
