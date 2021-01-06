from jax.lib import xla_bridge

if __name__ == "__main__":
    print(f"Number of GPUs found on system: {xla_bridge.device_count()}")
    xla_backend = xla_bridge.get_backend()
    xla_backend_type = xla_bridge.get_backend().platform  # cpu, gpu, tpu
    print(f"XLA backend type: {xla_backend_type}")
    if xla_backend_type == "gpu":
        for device in xla_backend.devices():
            print(f"Active GPU index: {device.id}")
            print(f"Active GPU name: {device.device_kind}")
