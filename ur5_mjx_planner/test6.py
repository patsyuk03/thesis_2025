import jax
import jax.numpy as jnp
import time

# Check available devices
devices = jax.devices()
print("Available devices:", devices)

# Create a large random matrix
size = 5000  # Large enough to see GPU benefits
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (size, size))

# Define a JIT-compiled matrix multiplication function
@jax.jit
def matmul(x):
    return jnp.dot(x, x)

# Run on CPU
cpu_device = jax.devices("cpu")[0]
x_cpu = jax.device_put(x, cpu_device)  # Move data to CPU
start_cpu = time.time()
y_cpu = matmul(x_cpu).block_until_ready()  # Ensure full execution
end_cpu = time.time()

# Run on GPU
gpu_device = jax.devices("gpu")[0] if len(jax.devices("gpu")) > 0 else None
if gpu_device:
    x_gpu = jax.device_put(x, gpu_device)  # Move data to GPU
    start_gpu = time.time()
    y_gpu = matmul(x_gpu).block_until_ready()  # Ensure full execution
    end_gpu = time.time()
else:
    print("No GPU detected. Skipping GPU test.")
    end_gpu = start_gpu = None

# Print results
print(f"CPU time: {end_cpu - start_cpu:.4f} seconds")
if gpu_device:
    print(f"GPU time: {end_gpu - start_gpu:.4f} seconds")
