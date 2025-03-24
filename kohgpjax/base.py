from typing import Callable
from abc import abstractmethod

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxtyping import Float

import gpjax as gpx
from kohgpjax.kernel import KOHKernel
from kohgpjax.posterior import KOHPosterior, construct_posterior

class AbstractKOHModel:
    def __init__(self, x, t, y):
        self.x, self.t, self.y = x, t, y
        self.num_field_obs = self.y.shape[0] - self.t.shape[0]
        self.num_sim_obs = self.t.shape[0]

    ############## HANDLING DATA ##############
    ###########################################

    def dataset(self, theta) -> gpx.Dataset:
        """
        theta: jnp.ndarray
        """
        t = jnp.vstack((jnp.zeros((self.num_field_obs, self.t.shape[1])) + theta, self.t))
        x = jnp.hstack((self.x, t), dtype=jnp.float64)
        return gpx.Dataset(X=x, y=self.y)
    
    ############## GPJAX MODEL ##############
    #########################################

    def GP_prior_mean_function(self) -> gpx.mean_functions.AbstractMeanFunction:
        return gpx.mean_functions.Zero()
    

    def GP_prior(
        self, 
        prior_mean_function: gpx.mean_functions.AbstractMeanFunction,
        kernel: gpx.kernels.AbstractKernel
    ) -> gpx.gps.Prior:
        return gpx.gps.Prior(
            mean_function=prior_mean_function, 
            kernel=kernel,
            jitter=0.
        )
    
    @abstractmethod
    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_delta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_epsilon(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    def GP_kernel(
        self,
        GPJAX_params
    ) -> gpx.kernels.AbstractKernel:        
        return KOHKernel(
            num_field_obs = self.num_field_obs,
            num_sim_obs = self.num_sim_obs,
            k_eta = self.k_eta(GPJAX_params),
            k_delta = self.k_delta(GPJAX_params),
            k_epsilon = self.k_epsilon(GPJAX_params),
            k_epsilon_eta = self.k_epsilon_eta(GPJAX_params),
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
    
    def GP_posterior(
        self,
        GPJAX_params,
    ) -> KOHPosterior:
        prior = self.GP_prior(
            self.GP_prior_mean_function(), 
            self.GP_kernel(GPJAX_params)
        )
        likelihood = self.likelihood(
            num_datapoints=self.y.shape[0], 
            obs_stddev=jnp.array(0.0) # This is defined in the kernel, hence 0 here.
        )
        return construct_posterior(prior, likelihood)
    

    ############## KOH MODEL ##############
    #######################################

    @abstractmethod
    def KOH_log_prior(
        self,
        GPJAX_params,
    ) -> Float:
        raise NotImplementedError

    def get_KOH_neg_log_pos_dens_func(
            self,
            transform_params_to_GPJAX: Callable[[list], tuple[list, Float]] = lambda x: (x, 0)
    ) -> Callable[..., Float]:
        """Returns a function which calculates the negative log density of the model.
        Note the first parameter argument must be the calibration parameters.
        """
        def neg_log_dens(MCMC_params):
            GPJAX_params, log_det_jacobian = transform_params_to_GPJAX(MCMC_params)
            return self.objective()(
                self.GP_posterior(GPJAX_params), 
                self.dataset(jnp.array(GPJAX_params[0]))
            ) - self.KOH_log_prior(GPJAX_params) - log_det_jacobian
        return neg_log_dens
