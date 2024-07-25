from jaxtyping import Float

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gpjax as gpx
from kohgpjax.base import AbstractKOHModel

from MATLAB_mappings import ell2rho

class MatlabModel(AbstractKOHModel):
    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas = GPJAX_params
        return gpx.kernels.ProductKernel(
            kernels=[
                gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(ells[0]),
                    variance=jnp.array(1/lambdas[0])
                ), 
                gpx.kernels.RBF(
                    active_dims=[1],
                    lengthscale=jnp.array(ells[1]),
                )
            ]
        )
    
    def k_delta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambdas[1])
            )
    
    def k_epsilon(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambdas[2])
            )
    
    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambdas[3])
            )


    def KOH_log_prior(
        self,
        GPJAX_params,
    ) -> Float:
        # if lambda_epsilon_eta > 2e5 or lambda_epsilon_eta < 100.0:
        #     return -9e99

        thetas, ells, lambdas = GPJAX_params

        rho_eta_1 = ell2rho(ells[0])
        rho_eta_2 = ell2rho(ells[1])

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
        logprior += (10-1)*jnp.log(lambdas[0]) - 10*lambdas[0]

        # % Prior for lambda_b
        # % EXAMPLE: lambda_b ~ GAM(10,.3)
        # logprior = logprior + (10-1)*log(lambda_b) - .3*lambda_b;
        logprior += (10-1)*jnp.log(lambdas[1]) - 0.3*lambdas[1]

        # % Prior for lambda_e
        # % EXAMPLE: lambda_e ~ GAM(10,.001)
        # logprior = logprior + (10-1)*log(lambda_e) - .001*lambda_e;
        logprior += (10-1)*jnp.log(lambdas[2]) - 0.001*lambdas[2]

        # % Prior for lambda_en
        # % EXAMPLE: lambda_en ~ GAM(10,.001)
        # logprior = logprior + (10-1)*log(lambda_en) - .001*lambda_en;
        logprior += (10-1)*jnp.log(lambdas[3]) - 0.001*lambdas[3]

        return logprior