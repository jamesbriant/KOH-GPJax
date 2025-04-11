from abc import abstractmethod
from typing import Callable, Dict

from flax import nnx
import gpjax as gpx
from gpjax.typing import (
    ScalarFloat,
    # Array,
)
import jax
import jax.numpy as jnp
from jaxtyping import (
    Float, 
    # Num,
)


from kohgpjax.dataset import KOHDataset
from kohgpjax.gps import KOHPosterior, construct_posterior
from kohgpjax.kernels.kohkernel import KOHKernel
from kohgpjax.parameters import ModelParameters, SampleDict

class KOHModel(nnx.Module):
    """
    Class for a KOH model.
    """
    def __init__(
        self,
        model_parameters: ModelParameters,
        kohdataset: KOHDataset
    ):
        """
        Parameters:
        -----------
        model_parameters: CalibrationModelParameters
            The model parameters for the KOH model.
        kohdataset: KOHDataset
            The dataset containing the field and simulation observations.
        """
        self.model_parameters = model_parameters
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
    def k_eta(self, GPJAX_params: SampleDict) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_delta(self, GPJAX_params: SampleDict) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_epsilon(self, GPJAX_params: SampleDict) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError
    
    @abstractmethod
    def k_epsilon_eta(self, GPJAX_params: SampleDict) -> gpx.kernels.AbstractKernel:
        raise NotImplementedError # TODO: Should this change to a constant 0 by default? White noise?
    
    def GP_kernel(
        self,
        GPJAX_params: Dict[str, SampleDict]
    ) -> gpx.kernels.AbstractKernel:        
        return KOHKernel(
            num_field_obs = self.kohdataset.num_field_obs,
            num_sim_obs = self.kohdataset.num_sim_obs,
            k_eta = self.k_eta(GPJAX_params['eta']),
            k_delta = self.k_delta(GPJAX_params['delta']),
            k_epsilon = self.k_epsilon(GPJAX_params['epsilon']),
            k_epsilon_eta = self.k_epsilon_eta(GPJAX_params['epsilon_eta']),
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

    def get_KOH_neg_log_pos_dens_func(self) -> Callable[..., Float]:
        """Returns a function which calculates the negative log density of the model.
        Note the first parameter argument must be the calibration parameters.
        """
        neg_log_like_func = gpx.objectives.ConjugateMLL(
            negative=True
        )
        log_prior_func = self.model_parameters.get_log_prior_func()

        def neg_log_dens(params_unconstrained_flat) -> Float:
            params_unconstrained_flat_list = [x for x in params_unconstrained_flat]
            params_constrained = self.model_parameters.constrain_and_unflatten_sample(params_unconstrained_flat)

            # thetas_list needs to maintain the correct order!
            thetas_list = [params_constrained['thetas'][f"theta_{i}"] for i in range(self.kohdataset.num_calib_params)]
            dataset = self.kohdataset.get_dataset(jnp.array(thetas_list).reshape(-1,1))

            neg_log_like = neg_log_like_func(
                self.GP_posterior(params_constrained),
                dataset
            )
            log_prior = log_prior_func(params_unconstrained_flat_list)

            return neg_log_like - log_prior
        
        return neg_log_dens
