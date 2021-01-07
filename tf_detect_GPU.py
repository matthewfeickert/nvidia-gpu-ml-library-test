import tensorflow as tf
from tensorflow.config.experimental import get_device_details

if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices("GPU")
    # Getting device details now to avoid info stream later during prints
    gpu_details = [get_device_details(gpu) for gpu in gpu_devices]

    print(f"\nTensorFlow build supports GPU: {tf.test.is_built_with_gpu_support()}")
    print(f"TensorFlow build supports XLA: {tf.test.is_built_with_xla()}")
    print(f"TensorFlow build supports CUDA: {tf.test.is_built_with_cuda()}")

    for device, details in zip(gpu_devices, gpu_details):
        print(f"\nGPU: {device.name}")
        print(f"GPU index: {device.name.split(':')[-1]}")
        print(f"GPU name: {details['device_name']}")
