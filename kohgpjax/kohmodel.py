from abc import abstractmethod
from typing import Callable

import beartype.typing as tp
import gpjax as gpx
import jax.numpy as jnp
import jaxtyping  # Changed from 'from jaxtyping import Float'
from flax import nnx
from gpjax.typing import Array  # Added
from jax.experimental import checkify

from kohgpjax.dataset import KOHDataset
from kohgpjax.gps import KOHPosterior, construct_posterior
from kohgpjax.kernels.kohkernel import KOHKernel
from kohgpjax.parameters import (
    ModelParameterDict,
    ModelParameters,  # noqa: F401
)

MP = tp.TypeVar("MP", bound="ModelParameters")


class KOHModel(nnx.Module):
    """
    Class for a KOH model.
    """

    def __init__(
        self,
        model_parameters: MP,
        kohdataset: KOHDataset,
        obs_stddev: gpx.parameters.Static = None,
        jitter: float = 1e-6,
    ):
        """
        Parameters:
        -----------
        model_parameters: ModelParameters
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
        variance_to_use = self.obs_var
        if variance_to_use is None:
            # If obs_var is not static (i.e., learnable), get it from params_constrained
            # Assuming it's structured like: params_constrained['epsilon']['variances']['obs_noise']
            # This path needs to be consistent with how ModelParameters structures things.
            # The fixture `simple_prior_dict` has:
            # "epsilon": {"variances": {"obs_noise": ParameterPrior(npd.HalfNormal(1.0))}},
            variance_to_use = (
                params_constrained.get("epsilon", {})
                .get("variances", {})
                .get("obs_noise")
            )
            if variance_to_use is None:
                raise ValueError(
                    "k_epsilon: variance is None from both self.obs_var and params_constrained."
                )

        return gpx.kernels.White(
            active_dims=list(range(self.kohdataset.num_variable_params)),
            variance=variance_to_use,
        )

    def k_epsilon_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.White(variance=0.0)

    def GP_kernel(self, GPJAX_params: ModelParameterDict) -> gpx.kernels.AbstractKernel:
        return KOHKernel(
            num_field_obs=self.kohdataset.num_field_obs,
            num_sim_obs=self.kohdataset.num_sim_obs,
            k_eta=self.k_eta(GPJAX_params),
            k_delta=self.k_delta(GPJAX_params),
            k_epsilon=self.k_epsilon(GPJAX_params),
            k_epsilon_eta=self.k_epsilon_eta(GPJAX_params),
        )

    def likelihood(
        self, num_datapoints: int, GPJAX_params: ModelParameterDict
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
        GPJAX_params: ModelParameterDict,
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

    def get_KOH_neg_log_pos_dens_func(
        self,
    ) -> Callable[..., jaxtyping.Scalar]:
        """Returns a function which calculates the negative log posterior density of the model."""
        log_like_func = gpx.objectives.conjugate_mll
        # Type of log_prior_func: Callable[[List[jaxtyping.Float[Array, "..."]]], jaxtyping.Float[Array, ""]]
        log_prior_func = self.model_parameters.get_log_prior_func()

        def neg_log_pos_dens(
            params_unconstrained_flat: jaxtyping.Float[Array, "..."],
        ) -> jaxtyping.Scalar:  # Assuming flat array input
            # log_prior_func expects a List of arrays/scalars.
            # If params_unconstrained_flat is a single flat array, it needs to be converted to List[Scalar JAX array]
            # This was handled in ModelParameters by tree_map. Here, let's assume params_unconstrained_flat is already a list
            # or that log_prior_func can handle a flat array if ModelParameters was updated.
            # Based on ModelParameters.get_log_prior_func, its argument is List[jaxtyping.Float[Array, "..."]]
            # So, params_unconstrained_flat_list should be this type.
            # The input to neg_log_pos_dens is often a flat JAX array from optimizers/samplers.
            params_unconstrained_flat_list = [
                x for x in params_unconstrained_flat
            ]  # This makes it a list of scalar arrays if input is 1D array

            params_constrained = self.model_parameters.constrain_and_unflatten_sample(
                params_unconstrained_flat  # ModelParameters methods take flat JAX array or list
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

        def nlpd_checkified(
            params_unconstrained_flat: jaxtyping.Float[Array, "..."],
        ) -> jaxtyping.Scalar:
            error, value = checkify.checkify(neg_log_pos_dens)(
                params_unconstrained_flat
            )
            return value

        return nlpd_checkified
