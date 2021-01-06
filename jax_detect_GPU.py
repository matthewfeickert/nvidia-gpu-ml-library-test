from jax.lib import xla_bridge

if __name__ == "__main__":
    xla_backend = xla_bridge.get_backend().platform  # cpu, gpu, tpu
    print(f"XLA backend: {xla_backend}")
