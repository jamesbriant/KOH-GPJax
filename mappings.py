from jax import numpy as jnp
from jaxtyping import Float

def mapRto01(omega) -> Float:
    return 1/(1+jnp.exp(omega))

def map01toR(theta) -> Float:
    # return jnp.log(-1+1/theta)
    return jnp.log(1-theta) - jnp.log(theta)

def mapRto0inf(omega) -> Float:
    return jnp.exp(omega)

def map0inftoR(theta) -> Float:
    return jnp.log(theta)

def ell2rho(l) -> Float:
    return jnp.exp(-1/(8*l**2))

def beta2ell(beta) -> Float:
    return 1/jnp.sqrt(2*beta)