import tensorflow as tf
from tensorflow.config.experimental import get_device_details

if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices("GPU")
    # Getting device details now to avoid info stream later during prints
    gpu_details = [get_device_details(gpu) for gpu in gpu_devices]

    print(f"\nTensorFlow build supports GPU: {tf.test.is_built_with_gpu_support()}")
    print(f"TensorFlow build supports XLA: {tf.test.is_built_with_xla()}")
    is_built_with_cuda = tf.test.is_built_with_cuda()
    print(f"TensorFlow build supports CUDA: {is_built_with_cuda}")

    if is_built_with_cuda:
        build_info = tf.sysconfig.get_build_info()
        print(f"TensorFlow build CUDA version: {build_info['cuda_version']}")
        print(f"TensorFlow build cuDNN version: {build_info['cudnn_version']}")

    for device, details in zip(gpu_devices, gpu_details):
        print(f"\nGPU: {device.name}")
        print(f"GPU index: {device.name.split(':')[-1]}")
        print(f"GPU name: {details['device_name']}")
