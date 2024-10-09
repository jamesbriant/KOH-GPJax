from jax import numpy as jnp
from jaxtyping import Float

def ell2rho(l) -> Float:
    return jnp.exp(-1/(8*l**2))

def rho2ell(rho) -> Float:
    return 1/jnp.sqrt(-8*jnp.log(rho))

def beta2ell(beta) -> Float:
    return 1/jnp.sqrt(2*beta)

def ell2beta(l) -> Float:
    return 1/(2*l**2)