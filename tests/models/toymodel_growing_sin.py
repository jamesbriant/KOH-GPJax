from jaxtyping import Float

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
# from jax import lax

import gpjax as gpx
from kohgpjax.base import AbstractKOHModel


class Model(AbstractKOHModel):
    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas, periods = GPJAX_params
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
        thetas, ells, lambdas, periods = GPJAX_params
        return gpx.kernels.SumKernel(
            kernels=[
                gpx.kernels.Periodic(
                    active_dims=[0],
                    lengthscale=jnp.array(ells[2]),
                    variance=jnp.array(1/lambdas[1]),
                    period=jnp.array(periods[0])
                ),
                gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(ells[3]),
                    variance=jnp.array(1/lambdas[4])
                )
            ]
        )
    
    def k_epsilon(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas, periods = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambdas[2])
            )
    
    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        thetas, ells, lambdas, periods = GPJAX_params
        return gpx.kernels.White(
                active_dims=[0],
                variance=jnp.array(1/lambdas[3])
            )


    def KOH_log_prior(
        self,
        GPJAX_params,
    ) -> Float:
        thetas, ells, lambdas, periods = GPJAX_params

        ####### ell #######
        # % Prior for ell_eta
        # % EXAMPLE: ell_eta_0 ~ GAM(2,1)
        logprior = (2-1)*jnp.log(ells[0]) - 1*ells[0]

        # % Prior for ell_eta_1 ~ GAM(3,0.5)
        logprior += (3-1)*jnp.log(ells[1]) - 0.5*ells[1]

        # # % Prior for ell_delta_0 ~ GAM(3,3)
        # logprior += (3-1)*jnp.log(ells[2]) - 3*ells[2]
        # # lax.cond(ells[2] < 10, lambda x: x, lambda x: x+9e99, logprior)
        # # % Prior for ell_delta_0 ~ GAM(2,1)
        logprior += (2-1)*jnp.log(ells[2]) - 1*ells[2]

        # % Prior for ell_delta_1 ~ GAM(2,1)
        logprior += (2-1)*jnp.log(ells[3]) - 1*ells[3]

        ####### lambda #######
        # % Prior for lambda_eta
        # % EXAMPLE: lambda_eta ~ GAM(2,1)
        logprior += (2-1)*jnp.log(lambdas[0]) - 1*lambdas[0]

        # % Prior for lambda_b
        # % EXAMPLE: lambda_b ~ GAM(10,.33)
        logprior += (10-1)*jnp.log(lambdas[1]) - 0.33*lambdas[1]

        # % Prior for lambda_e
        # % EXAMPLE: lambda_e ~ GAM(12,0.025)
        logprior += (12-1)*jnp.log(lambdas[2]) - 0.025*lambdas[2]

        # % Prior for lambda_en
        # % EXAMPLE: lambda_en ~ GAM(10,.001)
        logprior += (10-1)*jnp.log(lambdas[3]) - 0.001*lambdas[3]

        # # % Prior for lambda_delta
        # # % EXAMPLE: lambda_delta ~ GAM(10,0.33)
        logprior += (10-1)*jnp.log(lambdas[4]) - 0.33*lambdas[4]

        ####### period #######
        # % Prior for period_delta
        # % EXAMPLE: period_delta ~ GAM(2,1)
        logprior += (2-1)*jnp.log(periods[0]) - 1*periods[0]

        return logprior