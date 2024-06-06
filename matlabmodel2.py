# This is the same as MATLAB code but with the rho_delta_1 parameter removed
# as it is not used in the kernel.

# This is a major object orientated rewrite of matlabmodel.py

# from functools import partial
from typing import Any, Tuple, Callable

from gpjax.typing import Array
from jaxtyping import Float

import numpy as np

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
# from jax import random
# from jax import jit, grad

import gpjax as gpx
from kohgpjax.kernel import KOHKernel
from kohgpjax.posterior import *

# from mappings import mapRto01, mapRto0inf, ell2rho
from mappings import *


class MatlabModel:
    def __init__(self, x, t, y):
        self.x, self.t, self.y = x, t, y
        self.num_field_obs = self.y.shape[0] - self.t.shape[0]
        self.num_sim_obs = self.t.shape[0]

    def dataset(self, params) -> gpx.Dataset:
        theta, *_ = self._params_to_kernel_params(params)
        t = jnp.vstack((jnp.zeros((self.num_field_obs, self.t.shape[1])) + theta, self.t))
        x = jnp.hstack((self.x, t), dtype=np.float64)
        return gpx.Dataset(X=x, y=self.y)
    

    def prior_mean_function(self) -> gpx.mean_functions.AbstractMeanFunction:
        return gpx.mean_functions.Zero()
    

    def prior(
        self, 
        prior_mean_function: gpx.mean_functions.AbstractMeanFunction,
        kernel: gpx.kernels.AbstractKernel
    ) -> gpx.gps.Prior:
        return gpx.gps.Prior(
            mean_function=prior_mean_function, 
            kernel=kernel,
            jitter=0.
        )
    

    def kernel(self, params) -> gpx.kernels.AbstractKernel:
        theta, ell_eta_1, ell_eta_2, ell_delta_1, lambda_eta, lambda_delta, lambda_epsilon, lambda_epsilon_eta = self._params_to_kernel_params(params)

        product_kernel = gpx.kernels.ProductKernel(
            kernels=[
                gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(ell_eta_1),
                    variance=jnp.array(1/lambda_eta)
                ), 
                gpx.kernels.RBF(
                    active_dims=[1],
                    lengthscale=jnp.array(ell_eta_2)
                )
            ]
        )

        return KOHKernel(
            num_field_obs=self.num_field_obs,
            num_sim_obs=self.num_sim_obs,
            k_eta=product_kernel,
            # k_delta=gpx.kernels.White(
            #     active_dims=[0],
            #     variance=jnp.array(1/lambda_delta)
            # ),
            k_delta=gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(ell_delta_1),
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
    

    def likelihood(
        self,
        num_datapoints, 
        obs_stddev=jnp.array(0.0)
    ) -> gpx.likelihoods.AbstractLikelihood:
        return gpx.likelihoods.Gaussian(
            num_datapoints=num_datapoints,
            obs_stddev=obs_stddev
        )
    

    def objective(self) -> gpx.objectives.AbstractObjective:
        return gpx.objectives.ConjugateMLL(
            negative=True
        )


    def get_neg_log_dens_func(self) -> Callable[..., Float]:
        return lambda params: self.objective()(self.posterior(params), self.dataset(params)) - self.log_prior(params)


    def _params_to_kernel_params(self, params):
        theta = mapRto01(params[0])
        ell_eta_1 = mapRto0inf(params[1])
        ell_eta_2 = mapRto0inf(params[2])
        ell_delta_1 = mapRto0inf(params[3])
        lambda_eta = mapRto0inf(params[4])
        lambda_delta = mapRto0inf(params[5])
        lambda_epsilon = mapRto0inf(params[6])
        lambda_epsilon_eta = mapRto0inf(params[7])
        return theta, ell_eta_1, ell_eta_2, ell_delta_1, lambda_eta, lambda_delta, lambda_epsilon, lambda_epsilon_eta


    def posterior(
        self,
        params,
    ) -> KOHPosterior:
        prior = self.prior(
            self.prior_mean_function(), 
            self.kernel(params)
        )
        likelihood = self.likelihood(
            num_datapoints=self.y.shape[0], 
            obs_stddev=jnp.array(0.0) # This is defined in the kernel, hence 0 here.
        )
        return construct_posterior(prior, likelihood)


    def log_prior(
        self,
        params,
    ) -> Float:
        # if lambda_epsilon_eta > 2e5 or lambda_epsilon_eta < 100.0:
        #     return -9e99

        theta, ell_eta_1, ell_eta_2, ell_delta_1, lambda_eta, lambda_delta, lambda_epsilon, lambda_epsilon_eta = self._params_to_kernel_params(params)
        rho_eta_1 = ell2rho(ell_eta_1)
        rho_eta_2 = ell2rho(ell_eta_2)
        rho_delta_1 = ell2rho(ell_delta_1)

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
        logprior += -0.6*jnp.log(1-rho_delta_1)

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




# from dataloader import DataLoader
# dataloader = DataLoader()
# data = dataloader.get_data() # loads normalised/standardised data
# model = MatlabModel(*data)
# model_parameters = np.array([
#     map01toR(0.4257), 
#     map0inftoR(beta2ell(51.5551)), #these are the beta values!!!
#     map0inftoR(beta2ell(3.5455)), 
#     map0inftoR(beta2ell(2)), 
#     map0inftoR(0.25557), 
#     map0inftoR(37.0552), 
#     map0inftoR(10030.5142), 
#     map0inftoR(79548.2126)
# ])

# print(model.get_neg_log_dens_func()(model_parameters))

# neg_log_dens = jit(model.get_neg_log_dens_func())
# print(neg_log_dens(model_parameters))

# grad_neg_log_dens = jit(grad(neg_log_dens, argnums=0))
# print(grad_neg_log_dens(model_parameters))





# xpred = jnp.linspace(0, 1, 1000, endpoint=True)

# gpjax_posterior = model.posterior(model_parameters)
# gpjax_eta_pred = gpjax_posterior.predict_obs(
#     np.vstack((xpred, model_parameters[0]*np.ones_like(xpred))).T,
#     model.dataset(model_parameters)
# )

# m = gpjax_eta_pred.mean()
# v = gpjax_eta_pred.variance()
# sd = jnp.sqrt(v)