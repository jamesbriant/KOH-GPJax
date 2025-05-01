import gpjax as gpx
import jax.numpy as jnp
from jax import jit
from jax.experimental import checkify

print("GPJax version:", gpx.__version__)

x = jnp.array([[1.0], [2.0], [3.0]])

def get_kernel_gram_func(lengthscale):
    k = gpx.kernels.RBF(active_dims=[0], lengthscale=lengthscale, variance=jnp.array(1.0))
    return k.gram(x)

kgf = checkify.checkify(get_kernel_gram_func)
kgf_jit = jit(checkify.checkify(get_kernel_gram_func))

l = jnp.array([3.0])

print(kgf(l)[1].to_dense())
print(kgf_jit(l)[1].to_dense())