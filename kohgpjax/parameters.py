import jax
import jax.numpy as jnp
from jaxtyping import Float
import numpyro.distributions as npd
import numpyro.distributions.transforms as npt
from typing import Callable, Dict, List, Union

class ParameterPrior:
    def __init__(
        self,
        distribution: npd.Distribution,
        name: str = None,
    ):
        """Distribution on the constrained parameter.
        Args:
            distribution: A numpyro Distribution object representing the prior distribution.
            name: Optional name for the prior.
        """
        if not isinstance(distribution, npd.Distribution):
            raise ValueError("distribution must be a numpyro Distribution object.")
        
        self.distribution = distribution
        self.bijector = npt.biject_to(self.distribution.support) # This maps Reals to the constrained space
        self.name = name

    def forward(self, y):
        """Transform the input to the constrained space.
        Args:
            y: The unconstrained input value.
        """
        return self.bijector(y)
    
    def inverse(self, x):
        """Transform the input to the unconstrained space.
        Args:
            x: The constrained input value.
        """
        return self.bijector._inverse(x)
    
    def prob(self, y):
        """Compute the probability density function (PDF) of the distribution.
        Args:
            y: The unconstrained input value.
        """
        return jnp.exp(self.log_prob(y))
    
    def log_prob(self, y):
        """Compute the log probability density function (PDF) of the distribution.
        Args:
            y: The unconstrained input value.
        """
        x = self.bijector(y) # map y in Reals onto x in the constrained space
        logdet = self.bijector.log_abs_det_jacobian(y, x)

        return self.distribution.log_prob(x) + logdet
    
    def __repr__(self) -> str:
        repr = (
            f"Prior(\n"
            f"  distribution={self.distribution},\n"
            f"  bijector={self.bijector},\n"
            f"  name={self.name},\n"
            f")"
        )
        return repr


KernelParamsDict = Dict[
    str, Union[
        ParameterPrior,
        Dict[str, ParameterPrior],
    ]
]

PriorDict = Dict[
    str, KernelParamsDict
]

SampleDict = Dict[str, Float] #TODO: Is this correct? It should be a tree of samples, not just a dict of floats.

class ModelParameters:
    def __init__(self, prior_dict: PriorDict):
        """
        Initialize the calibration model parameters.
        :param kernel_config: A dictionary containing the kernel configuration.
        """
        # Check if the kernel config is valid 
        _check_prior_dict(prior_dict)
        #TODO: Check kernel_config structure agrees with the KOHDataset. Add this check to KOHPosterior?
        # Things to check:
        # - Dimensions of the kernel match the dimensions of the dataset
        # - There are enough theta priors
        
        self.priors = prior_dict

        self.priors_flat, self.priors_tree = jax.tree.flatten(self.priors)
        self.n_params = len(self.priors_flat)

        self.prior_log_prob_funcs: List[Callable[[Float], Float]] = jax.tree.map(
            lambda dist: jax.jit(dist.log_prob),
            self.priors_flat
        )
    
    def constrain_sample(self, samples_flat):
        """
        Transform samples to the constrained space.
        Args:
            samples_flat: A flat JAX array of samples.
        Returns:
            A tree of samples in the constrained space.
        """
        return [prior.forward(samples_flat[i]) for i, prior in enumerate(self.priors_flat)]

    def unflatten_sample(self, samples_flat) -> SampleDict:
        """
        Unflatten the samples to the original prior tree structure.
        Args:
            samples_flat: A flat array of samples.
        Returns:
            A tree of samples with the same structure as the priors.
        """
        # Unflatten the samples to the original tree structure
        return jax.tree.unflatten(self.priors_tree, samples_flat)
    
    def constrain_and_unflatten_sample(self, samples_flat) -> SampleDict:
        """
        Transform samples to the constrained space and unflatten them to the original prior tree structure.
        Args:
            samples_flat: A flat array of samples.
        Returns:
            A tree of samples in the constrained space with the same structure as the priors.
        """
        # Constrain and unflatten the samples
        constrained_samples = self.constrain_sample(samples_flat)
        return self.unflatten_sample(constrained_samples)


    def get_log_prior_func(self) -> Callable[[List], Float]:
        """Compute the joint log prior probability.
    
        Returns:
            A function that computes the joint log prior probability.
        """
        @jax.jit
        def log_prior_func(unconstrained_params_flat: List) -> Float:
            """
            Compute the joint log prior probability.
            Args:
                unconstrained_params_flat: A flat array of unconstrained parameters.
            Returns:
                The joint log prior probability.
            """
            log_probs = jax.tree.map(
                lambda log_prob_func, x: log_prob_func(x),
                self.prior_log_prob_funcs,
                unconstrained_params_flat
            )
            return jnp.sum(jnp.concatenate([jnp.atleast_1d(x) for x in log_probs]))
        
        return log_prior_func
    

def _check_prior_dict(prior_dict: PriorDict):
    """
    Check if the kernel config is valid.
    """
    # Check if the kernel config has the correct keys
    #TODO: Make epsilon optional?
    # if not set(['thetas', 'eta', 'delta', 'epsilon']).issubset(list(prior_dict.keys())):
    #     raise ValueError("prior_dict keys must contain ['thetas', 'eta', 'delta', 'epsilon']")
    required_keys = ['thetas', 'eta', 'delta'] # 'epsilon' and 'epsilon_eta' are optional
    if not set(required_keys).issubset(set(prior_dict.keys())):
        raise ValueError(f"prior_dict keys must contain {required_keys}")
    
    # Check if each kernel has the required keys
    for key, param_prior_dict in prior_dict.items():
        if key == 'thetas':
            if not isinstance(param_prior_dict, dict):
                raise ValueError(f"prior_dict['{key}'] must be a dictionary of ParameterPrior instances.")
            for sub_param_name, sub_param_item in param_prior_dict.items():
                if not isinstance(sub_param_item, ParameterPrior):
                    raise ValueError(f"prior_dict['thetas']['{sub_param_name}'] must be a ParameterPrior instance.")
                
        else:
            if 'variances' not in param_prior_dict:
                raise KeyError(f"prior_dict key '{key}' must contain 'variances' key with dictionary of ParameterPrior instances.")

            for sub_param_type, sub_param_dict in param_prior_dict.items():
                if not isinstance(sub_param_dict, dict):
                    raise ValueError(f"prior_dict['{key}']['{sub_param_type}'] must be a dictionary of ParameterPrior instances.")
                
                for sub_param_type_name, param_prior in sub_param_dict.items():
                    if not isinstance(param_prior, ParameterPrior):
                        raise ValueError(f"prior_dict['{key}']['{sub_param_type}']['{sub_param_type_name}'] must be a ParameterPrior instance.")
        
        #TODO: Add checks to ensure prior_dict is compatible with the KOHModel() instance.

        #TODO: add checks for parameters depending on the AbstractKernel class.
        # e.g. if kernel_item['kernel'] is RBF, check that lengthscale is a npd.Distribution
        # How to get the parameters from the kernel class?
        # Is this worth the faff?
