import tensorflow as tf

# from tensorflow.python.client import device_lib
from tensorflow.config.experimental import get_device_details

if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_details = [get_device_details(gpu) for gpu in gpu_devices]
    # device_list = device_lib.list_local_devices()

    print("\n\n\n")
    print(f"TensorFlow build supports GPU: {tf.test.is_built_with_gpu_support()}")
    print(f"TensorFlow build supports XLA: {tf.test.is_built_with_xla()}")
    print(f"TensorFlow build supports CUDA: {tf.test.is_built_with_cuda()}")

    for device, details in zip(gpu_devices, gpu_details):
        print(f"\nGPU: {device.name}")
        print(f"GPU name: {details['device_name']}")

    # for device in device_list:
    #     if device.device_type == "GPU":
    #         print(device.physical_device_desc)

    # >>> gpu_devices = tf.config.list_physical_devices('GPU')
    # >>> if gpu_devices:
    # ...   details = tf.config.experimental.get_device_details(gpu_devices[0])
    # ...   details.get('device_name', 'Unknown GPU')
