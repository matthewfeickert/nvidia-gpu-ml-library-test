from jax.lib import xla_bridge

if __name__ == "__main__":
    xla_backend = xla_bridge.get_backend()
    xla_backend_type = xla_bridge.get_backend().platform  # cpu, gpu, tpu
    print(f"XLA backend type: {xla_backend_type}")

    gpu_count = xla_bridge.device_count() if xla_backend_type == "gpu" else 0
    print(f"\nNumber of GPUs found on system: {gpu_count}")
    if xla_backend_type == "gpu":
        for idx, device in enumerate(xla_backend.devices()):
            gpu_type = "Active GPU" if idx == 0 else "GPU"
            print(f"\n{gpu_type} index: {device.id}")
            print(f"{gpu_type} name: {device.device_kind}")
