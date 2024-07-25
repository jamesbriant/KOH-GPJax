from jaxtyping import Float

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gpjax as gpx
from kohgpjax.base import AbstractKOHModel

from MATLAB_mappings import ell2rho

class MatlabModel(AbstractKOHModel):
    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        theta, ell_eta_1, ell_eta_2, lambda_eta, *_ = GPJAX_params
        return gpx.kernels.ProductKernel(
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
    
    def k_delta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        _, _, _, _, lambda_delta, _, _ = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambda_delta)
            )
    
    def k_epsilon(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        _, _, _, _, _, lambda_epsilon, _ = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambda_epsilon)
            )
    
    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        _, _, _, _, _, _, lambda_epsilon_eta = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambda_epsilon_eta)
            )


    def KOH_log_prior(
        self,
        GPJAX_params,
    ) -> Float:
        # if lambda_epsilon_eta > 2e5 or lambda_epsilon_eta < 100.0:
        #     return -9e99

        theta, ell_eta_1, ell_eta_2, lambda_eta, lambda_delta, lambda_epsilon, lambda_epsilon_eta = GPJAX_params

        rho_eta_1 = ell2rho(ell_eta_1)
        rho_eta_2 = ell2rho(ell_eta_2)

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