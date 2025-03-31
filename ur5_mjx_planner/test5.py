import jax.numpy as jnp
from jax import jit
import jax
import time

# Check available devices
print("Available devices:", jax.devices())

# Define a simple function
def slow_f(x):
    return x * x + x * 2.0

# Compile with JIT for better GPU performance
fast_f = jit(slow_f)

# Create input and explicitly move it to GPU
x = jnp.ones((5000, 5000))
x = jax.device_put(x, jax.devices("gpu")[0])  # Move data to GPU

# Measure JIT-compiled function execution time
start_fast = time.time()
y = fast_f(x).block_until_ready()  # Ensure full execution before timing
dfast = time.time() - start_fast

# Measure non-JIT function execution time
start_slow = time.time()
y_slow = slow_f(x).block_until_ready()  # Ensure full execution
dslow = time.time() - start_slow

print(f"JIT-compiled (GPU) time: {dfast * 1000:.2f} ms")
print(f"Non-JIT (GPU) time: {dslow * 1000:.2f} ms")
