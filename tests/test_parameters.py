from jax import config
config.update("jax_enable_x64", True)

from gpjax.kernels import ProductKernel, RBF, White
from gpjax.kernels.base import AbstractKernel, CombinationKernel
from jax import numpy as jnp
from jax import jit
import numpyro.distributions as npd
from typing import Union, List

from kohgpjax.parameters import (
    ParameterPrior,
    ModelParameters,
    PriorDict,
)

prior_dict: PriorDict = {
    'eta': {
        'parameters': {
            'thetas': {
                'theta_0': ParameterPrior(
                    npd.Normal(loc=0.0, scale=1.0),
                ),
            },
            'variance': ParameterPrior(
                npd.Normal(loc=0.0, scale=0.5), 
            ),
            'lengthscale': {
                'x_0': ParameterPrior(
                    npd.Normal(loc=0.0, scale=1.0),
                ),
                'theta_0': ParameterPrior(
                    npd.Normal(loc=0.0, scale=1.0),
                ),
            },
        },
    },
    'discrepancy': {
        'parameters': {
            'variance': ParameterPrior(
                npd.Normal(loc=0.0, scale=0.5),
            ),
            'lengthscale': {
                'x_0': ParameterPrior(
                    npd.Normal(loc=0.0, scale=1.0),
                ),
            },
        },
    },
    'epsilon': {
        'parameters': {
            'variance': ParameterPrior(
                npd.Normal(loc=0.0, scale=0.5),
            ),
        },
    },
    'epsilon_eta': { #TODO: make this optional
        'parameters': {
            'variance': ParameterPrior(
                npd.Normal(loc=0.1, scale=0.5),
            ),
        },
    },
}

params = ModelParameters(prior_dict)

print(params.priors['eta']['variance'], '\n')
# Prior(
#   distribution=<distrax._src.distributions.normal.Normal object at 0x16c7c8bd0>,
#   bijector=<distrax._src.bijectors.inverse.Inverse object at 0x16c7b8d90>,
# )

log_prior = params.get_log_prior_func()
MCMC_sample = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
print(log_prior(MCMC_sample), '\n')
# -59.29095730955399

print(jit(params.constrain_sample)(MCMC_sample), '\n')
# [
#   Array(0., dtype=float64, weak_type=True), 
#   Array(0.76159416, dtype=float64, weak_type=True), 
#   Array(0.96402758, dtype=float64, weak_type=True), 
#   Array(0.99505475, dtype=float64, weak_type=True), 
#   Array(0.9993293, dtype=float64, weak_type=True), 
#   Array(0.9999092, dtype=float64, weak_type=True), 
#   Array(0.99998771, dtype=float64, weak_type=True), 
#   Array(0.99999834, dtype=float64, weak_type=True)
# ] 

print(jit(params.unflatten_sample)(MCMC_sample), '\n')
# {
#     'discrepancy': {
#         'lengthscale': {
#             'x_0': 0.0
#         }, 
#         'variance': 1.0
#     }, 
#     'epsilon': {
#         'variance': 2.0
#     }, 
#     'epsilon_eta': {
#         'variance': 3.0
#     }, 
#     'eta': {
#         'lengthscale': {
#             'theta_0': 4.0, 
#             'x_0': 5.0
#         }, 
#         'thetas': {
#             'theta_0': 6.0
#         }, 
#         'variance': 7.0
#     }
# } 

print(jit(params.constrain_and_unflatten_sample)(MCMC_sample), '\n')
# {
#     'discrepancy': {
#         'lengthscale': {
#             'x_0': Array(0., dtype=float64, weak_type=True)
#         }, 
#         'variance': Array(0.76159416, dtype=float64, weak_type=True)
#     }, 
#     'epsilon': {
#         'variance': Array(0.96402758, dtype=float64, weak_type=True)
#     }, 
#     'epsilon_eta': {
#         'variance': Array(0.99505475, dtype=float64, weak_type=True)
#     }, 
#     'eta': {
#         'lengthscale': {
#             'theta_0': Array(0.9993293, dtype=float64, weak_type=True), 
#             'x_0': Array(0.9999092, dtype=float64, weak_type=True)
#         }, 
#         'thetas': {
#             'theta_0': Array(0.99998771, dtype=float64, weak_type=True)
#         }, 
#         'variance': Array(0.99999834, dtype=float64, weak_type=True)
#     }
# } 