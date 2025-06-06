from abc import abstractmethod
from typing import (
    Callable,
    Dict,
)

from flax import nnx
import gpjax as gpx
from jax.experimental import checkify
import jax.numpy as jnp
from jaxtyping import Float

from kohgpjax.dataset import KOHDataset
from kohgpjax.gps import (
    KOHPosterior,
    construct_posterior,
)
from kohgpjax.kernels.kohkernel import KOHKernel
from kohgpjax.parameters import ModelParameters, SampleDict


class KOHModel(nnx.Module):
    """
    Class for a KOH model.
    """

    def __init__(
        self,
        model_parameters: ModelParameters,
        kohdataset: KOHDataset,
        obs_stddev: gpx.parameters.Static = None,
        jitter: float = 1e-6,
    ):
        """
        Parameters:
        -----------
        model_parameters: CalibrationModelParameters
            The model parameters for the KOH model.
        kohdataset: KOHDataset
            The dataset containing the field and simulation observations.
        obs_stddev: gpx.parameters.Static
            The standard deviation of the observations. If not None, it will be static and not estimated.
        jitter: float
            The jitter to add to the covariance matrix for numerical stability.
        """
        if obs_stddev is not None:
            if not isinstance(obs_stddev, gpx.parameters.Static):
                raise ValueError(
                    "`obs_stddev` must be a `gpx.parameters.Static` object or `None`."
                )
            # if obs_stddev.shape != (1,): #TODO: This should be changed to allow a vector of variances
            #     raise ValueError("`obs_stddev` must have shape `(1,)`. For more complex models, implement your own `k_epsilon()` method.")
            self.obs_stddev = obs_stddev

        self.model_parameters = model_parameters
        self.kohdataset = kohdataset
        self.obs_var = (
            gpx.parameters.Static(obs_stddev**2) if obs_stddev is not None else None
        )
        self.jitter = jitter  # GPJax default is 1e-6

    ############## GPJAX MODEL ##############
    #########################################

    def GP_prior_mean_function(self) -> gpx.mean_functions.AbstractMeanFunction:
        return gpx.mean_functions.Zero()

    def GP_prior(
        self,
        prior_mean_function: gpx.mean_functions.AbstractMeanFunction,
        kernel: gpx.kernels.AbstractKernel,
    ) -> gpx.gps.Prior:
        return gpx.gps.Prior(
            mean_function=prior_mean_function,
            kernel=kernel,
            jitter=self.jitter,
        )

    @abstractmethod
    def k_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        """
        Returns the eta kernel, which is used to model the structure of the
        field observations and simulation outputs.
        To be implemented by subclasses.
        Args:
            params_constrained: The constrained parameters of the model.
        Returns:
            A GPJAX kernel.
        """
        raise NotImplementedError

    @abstractmethod
    def k_delta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        """
        Returns the delta kernel, which is used to model the structure of the
        calibration parameters.
        To be implemented by subclasses.
        Args:
            params_constrained: The constrained parameters of the model.
        Returns:
            A GPJAX kernel.
        """
        raise NotImplementedError

    def k_epsilon(self, params_constrained) -> gpx.kernels.AbstractKernel:
        """
        Returns the epsilon kernel, which defaults to a white noise kernel.
        This is used to model the observation variance.
        Args:
            params_constrained: The constrained parameters of the model.
        Returns:
            A GPJAX white noise kernel with the observation variance.
        """
        return gpx.kernels.White(
            active_dims=list(range(self.kohdataset.num_variable_params)),
            variance=self.obs_var,
        )

    def k_epsilon_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.White(variance=0.0)

    def GP_kernel(
        self, GPJAX_params: Dict[str, SampleDict]
    ) -> gpx.kernels.AbstractKernel:
        return KOHKernel(
            num_field_obs=self.kohdataset.num_field_obs,
            num_sim_obs=self.kohdataset.num_sim_obs,
            k_eta=self.k_eta(GPJAX_params),
            k_delta=self.k_delta(GPJAX_params),
            k_epsilon=self.k_epsilon(GPJAX_params),
            k_epsilon_eta=self.k_epsilon_eta(GPJAX_params),
        )

    def likelihood(
        self, num_datapoints: int, GPJAX_params: Dict[str, SampleDict]
    ) -> gpx.likelihoods.AbstractLikelihood:
        """
        Constructs the likelihood for the KOH model.
        Args:
            num_datapoints: The number of data points in the dataset.
            GPJAX_params: The GPJAX parameters in the same shape as prior_dict.
        Returns:
            A GPJAX likelihood object.
        """
        return gpx.likelihoods.Gaussian(
            num_datapoints=num_datapoints,
            obs_stddev=0.0,  # See self.k_epsilon()
        )

    def GP_posterior(
        self,
        GPJAX_params: Dict,
    ) -> KOHPosterior:
        """
        Constructs the GP posterior using the GPJAX parameters.
        Args:
            GPJAX_params: The GPJAX parameters in the same shape as prior_dict.
        Returns:
            A KOHPosterior object.
        """
        prior = self.GP_prior(
            self.GP_prior_mean_function(), self.GP_kernel(GPJAX_params)
        )
        likelihood = self.likelihood(
            num_datapoints=self.kohdataset.num_field_obs + self.kohdataset.num_sim_obs,
            GPJAX_params=GPJAX_params,
        )
        return construct_posterior(prior, likelihood)

    ############## KOH MODEL ##############
    #######################################

    def get_KOH_neg_log_pos_dens_func(self) -> Callable[..., Float]:
        """Returns a function which calculates the negative log posterior density of the model."""
        log_like_func = gpx.objectives.conjugate_mll
        log_prior_func = self.model_parameters.get_log_prior_func()

        def neg_log_pos_dens(params_unconstrained_flat) -> Float:
            params_unconstrained_flat_list = [x for x in params_unconstrained_flat]
            params_constrained = self.model_parameters.constrain_and_unflatten_sample(
                params_unconstrained_flat
            )

            # thetas_list needs to maintain the correct order!
            thetas_list = [
                params_constrained["thetas"][f"theta_{i}"]
                for i in range(self.kohdataset.num_calib_params)
            ]
            dataset = self.kohdataset.get_dataset(jnp.array(thetas_list).reshape(-1, 1))

            neg_log_like = -log_like_func(
                self.GP_posterior(params_constrained), dataset
            )
            log_prior = log_prior_func(params_unconstrained_flat_list)

            return neg_log_like - log_prior

        def nlpd_checkified(params_unconstrained_flat) -> Float:
            error, value = checkify.checkify(neg_log_pos_dens)(
                params_unconstrained_flat
            )
            return value

        return nlpd_checkified
