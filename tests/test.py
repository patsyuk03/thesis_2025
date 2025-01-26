import jax.numpy as jnp
from jax import vmap

def f(x, y):
    return x.sum() + y.sum()

xs = [1,2,3]
ys = jnp.array([[1,2,3],[1,2,3],[1,2,3]])
out = vmap(f, in_axes=(0,0))(xs, ys)
print(out)