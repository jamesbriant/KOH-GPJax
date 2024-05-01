# This is the same as MATLAB code but with the rho_delta_1 parameter removed
# as it is not used in the kernel.
from functools import partial
from typing import Any, Tuple, Callable

from gpjax.typing import Array
from jaxtyping import Float

import numpy as np

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import jit, grad

import gpjax as gpx
from kohgpjax.kernel import KOHKernel
from kohgpjax.posterior import *

from mappings import mapRto01, mapRto0inf, ell2rho

def load_data() -> Tuple[Array, Array, Array]:
    DATAFIELD = np.loadtxt('data/simple_field.csv', delimiter=',', dtype=np.float32)
    DATACOMP = np.loadtxt('data/simple_comp.csv', delimiter=',', dtype=np.float32)

    xf = np.reshape(DATAFIELD[:, 0], (-1, 1))
    xc = np.reshape(DATACOMP[:, 0], (-1,1))
    tc = np.reshape(DATACOMP[:, 1], (-1,1))
    yf = np.reshape(DATAFIELD[:, 1], (-1,1))
    yc = np.reshape(DATACOMP[:, 2], (-1,1))

    #Standardize full response using mean and std of yc
    yc_mean = np.mean(yc)
    yc_std = np.std(yc, ddof=1) #estimate is now unbiased
    x_min = min(xf.min(), xc.min())
    x_max = max(xf.max(), xc.max())
    t_min = tc.min()
    t_max = tc.max()

    xf_normalized = (xf - x_min)/(x_max - x_min)
    xc_normalized = (xc - x_min)/(x_max - x_min)
    # tc_normalized = np.zeros_like(tc)
    # for k in range(tc.shape[1]):
    #     tc_normalized[:, k] = (tc[:, k] - np.min(tc[:, k]))/(np.max(tc[:, k]) - np.min(tc[:, k]))
    tc_normalized = (tc - t_min)/(t_max - t_min)
    yc_standardized = (yc - yc_mean)/yc_std
    yf_standardized = (yf - yc_mean)/yc_std

    x_stack = jnp.vstack((xf_normalized, xc_normalized))
    y = jnp.vstack((yf_standardized, yc_standardized), dtype=np.float64)

    return x_stack, tc_normalized, y


def get_likelihood(
    num_datapoints, 
    obs_stddev=jnp.array(0.0)
) -> gpx.likelihoods.AbstractLikelihood:
    return gpx.likelihoods.Gaussian(
        num_datapoints=num_datapoints,
        obs_stddev=obs_stddev
    )

def get_objective() -> gpx.objectives.AbstractObjective:
    return gpx.objectives.ConjugateMLL(
        negative=True
    )

def get_prior_mean_function() -> gpx.mean_functions.AbstractMeanFunction:
    return gpx.mean_functions.Zero()

def log_prior(
        theta,
        rho_eta_1,
        rho_eta_2,
        # rho_delta_1,
        lambda_eta,
        lambda_delta,
        lambda_epsilon,
        lambda_epsilon_eta
) -> Float:
    # if lambda_epsilon_eta > 2e5 or lambda_epsilon_eta < 100.0:
    #     return -9e99

    ####### rho #######
    # % Prior for beta_eta
    # % rho_eta = exp(-beta_eta/4)
    # % EXAMPLE: rho_eta(k) ~ BETA(1,.5)
    # rho_eta = exp(-params.beta_eta/4);
    # rho_eta(rho_eta>0.999) = 0.999;
    # logprior = - .5*sum(log(1-rho_eta));
    logprior = -0.5*jnp.log(1-rho_eta_1)
    logprior += -0.5*jnp.log(1-rho_eta_2)
    # % rho_b = exp(-beta_b/4)
    # % EXAMPLE: rho_b(k) ~ BETA(1,.4) 
    # rho_b = exp(-params.beta_b/4);
    # rho_b(rho_b>0.999) = 0.999;
    # logprior = logprior - .6*sum(log(1-rho_b));
    # logprior += -0.6*jnp.log(1-rho_delta_1)

    ####### lambda #######
    # % Prior for lambda_eta
    # % EXAMPLE: lambda_eta ~ GAM(10,10)
    # logprior = logprior + (10-1)*log(lambda_eta) - 10*lambda_eta;
    logprior += (10-1)*jnp.log(lambda_eta) - 10*lambda_eta

    # % Prior for lambda_b
    # % EXAMPLE: lambda_b ~ GAM(10,.3)
    # logprior = logprior + (10-1)*log(lambda_b) - .3*lambda_b;
    logprior += (10-1)*jnp.log(lambda_delta) - 0.3*lambda_delta

    # % Prior for lambda_e
    # % EXAMPLE: lambda_e ~ GAM(10,.001)
    # logprior = logprior + (10-1)*log(lambda_e) - .001*lambda_e;
    logprior += (10-1)*jnp.log(lambda_epsilon) - 0.001*lambda_epsilon

    # % Prior for lambda_en
    # % EXAMPLE: lambda_en ~ GAM(10,.001)
    # logprior = logprior + (10-1)*log(lambda_en) - .001*lambda_en;
    logprior += (10-1)*jnp.log(lambda_epsilon_eta) - 0.001*lambda_epsilon_eta

    return logprior

def model(
    params,
    x_stack,
    tc_normalized,
    y,
    prior_mean_function,
    likelihood,
    objective,
):
    # print(theta)
    theta = mapRto01(params[0])
    ell_eta_1 = mapRto0inf(params[1])
    ell_eta_2 = mapRto0inf(params[2])
    # ell_delta_1 = mapRto0inf(params[3])
    lambda_eta = mapRto0inf(params[3])
    lambda_delta = mapRto0inf(params[4])
    lambda_epsilon = mapRto0inf(params[5])
    lambda_epsilon_eta = mapRto0inf(params[6])

    num_field_obs = y.shape[0] - tc_normalized.shape[0]
    num_sim_obs = tc_normalized.shape[0]

    t = jnp.vstack((jnp.zeros((num_field_obs, tc_normalized.shape[1])) + theta, tc_normalized))
    x = jnp.hstack((x_stack, t), dtype=np.float64)
    data = gpx.Dataset(X=x, y=y)
    
    product_kernel = gpx.kernels.ProductKernel(
        kernels=[
            gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(ell_eta_1),
                variance=jnp.array(1/lambda_eta)
            ), 
            gpx.kernels.RBF(
                active_dims=[1],
                lengthscale=jnp.array(ell_eta_2),
            )
        ]
    )

    kernel = KOHKernel(
        num_field_obs=num_field_obs,
        num_sim_obs=num_sim_obs,
        k_eta=product_kernel,
        k_delta=gpx.kernels.White(
            active_dims=[0],
            variance=jnp.array(1/lambda_delta)
        ), 
        k_epsilon=gpx.kernels.White(
            active_dims=[0],
            variance=jnp.array(1/lambda_epsilon)
        ),
        k_epsilon_eta=gpx.kernels.White(
            active_dims=[0],
            variance=jnp.array(1/lambda_epsilon_eta)
        ),
    )

    prior = gpx.gps.Prior(
        mean_function=prior_mean_function, 
        kernel=kernel,
        jitter=0.
    )
    posterior = construct_posterior(prior, likelihood)
    log_prior_dens = log_prior(
        theta,
        ell2rho(ell_eta_1),
        ell2rho(ell_eta_2),
        # ell2rho(ell_delta_1),
        lambda_eta,
        lambda_delta,
        lambda_epsilon,
        lambda_epsilon_eta
    )
    return objective(posterior, data) - log_prior_dens


def get_neg_log_dens() -> Callable[..., Float]:
    x_stack, tc_normalized, y = load_data()
    prior_mean_function = get_prior_mean_function()
    likelihood = get_likelihood(
        num_datapoints=y.shape[0], 
        obs_stddev=jnp.array(0.0)
    )
    objective = get_objective()

    return partial(
        model,
        x_stack=x_stack,
        tc_normalized=tc_normalized,
        y=y,
        prior_mean_function=prior_mean_function,
        likelihood=likelihood,
        objective=objective,
    )

def neg_log_dens(params: Array) -> Float:
    return get_neg_log_dens()(params)




# neg_log_dens = jit(get_neg_log_dens())
# grad_neg_log_dens = jit(grad(neg_log_dens, argnums=[0, 1, 2, 3, 4, 5, 6]))

# print(neg_log_dens(0.5, 50, 7, 1, 30, 1000, 10000))
# print(grad_neg_log_dens(0.5, 50., 7., 1., 30., 1000., 10000.))
# print(grad_neg_log_dens(0.5, 50., 7., 1., 30., 1000., 10000.)) # these should be the same as rho_delta_1 is not used in the kernel





# neg_log_dens = jit(get_neg_log_dens())
# grad_neg_log_dens = jit(grad(neg_log_dens, argnums=0))

# print(neg_log_dens([0.5, 50, 7, 30, 1, 30, 1000, 10000]))
# print(grad_neg_log_dens([0.5, 50., 7., 30., 1., 30., 1000., 10000.]))
# print(grad_neg_log_dens([0.5, 50., 7., 300., 1., 30., 1000., 10000.])) # these should be the same as rho_delta_1 is not used in the kernel