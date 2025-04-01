from abc import abstractmethod
from typing import Callable

from flax import nnx
import gpjax as gpx
from gpjax.typing import (
    ScalarFloat,
    # Array,
)
import jax.numpy as jnp
from jaxtyping import (
    Float, 
    # Num,
)

from kohgpjax.dataset import KOHDataset
from kohgpjax.gps import KOHPosterior, construct_posterior
from kohgpjax.kernels.kohkernel import KOHKernel

class AbstractKOHModel(nnx.Module):
    """
    Abstract class for a KOH model.
    """
    def __init__(
        self, 
        kohdataset: KOHDataset
    ):
        """
        Parameters:
        -----------
        kohdataset: KOHDataset
            The dataset containing the field and simulation observations.
        """
        self.kohdataset = kohdataset
    
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
        raise NotImplementedError # TODO: Should this change to a constant 0 by default? White noise?
    
    def GP_kernel(
        self,
        GPJAX_params
    ) -> gpx.kernels.AbstractKernel:        
        return KOHKernel(
            num_field_obs = self.kohdataset.num_field_obs,
            num_sim_obs = self.kohdataset.num_sim_obs,
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
    
    def GP_posterior(
        self,
        GPJAX_params,
    ) -> KOHPosterior:
        prior = self.GP_prior(
            self.GP_prior_mean_function(), 
            self.GP_kernel(GPJAX_params)
        )
        likelihood = self.likelihood(
            num_datapoints=self.kohdataset.num_field_obs + self.kohdataset.num_sim_obs, 
            obs_stddev=jnp.array(0.0) # This is defined in the kernel as field and sim are different, hence 0 here.
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
            transform_params_to_GPJAX: Callable[[list], list] = lambda x: x # TODO: Improve this workflow??
    ) -> Callable[..., Float]:
        """Returns a function which calculates the negative log density of the model.
        Note the first parameter argument must be the calibration parameters.
        """
        def neg_log_dens(MCMC_params):
            GPJAX_params = transform_params_to_GPJAX(MCMC_params)
            return gpx.objectives.ConjugateMLL(
                    negative=True
                )(
                    self.GP_posterior(GPJAX_params), 
                    # self.kohdataset.get_dataset(jnp.array(GPJAX_params[0]))
                    self.kohdataset.get_dataset(GPJAX_params[0])
                ) - self.KOH_log_prior(GPJAX_params)
        return neg_log_dens
