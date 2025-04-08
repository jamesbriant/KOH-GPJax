import distrax
from gpjax.kernels import AbstractKernel
import jax
import jax.numpy as jnp
from jaxtyping import Float
from typing import Callable, Dict, List, Union

class ParameterPrior:
    def __init__(
        self, 
        distribution: distrax.Distribution, 
        bijector: distrax.Bijector,
        invert_bijector: bool = False,
    ):
        """
        Args:
            distribution: A distrax.Distribution object representing the prior distribution.
            bijector: A distrax.Bijector object representing the transformation to the unconstrained space.
            invert_bijector: If True, the bijector is inverted.
        """
        if not isinstance(distribution, distrax.Distribution):
            raise ValueError("distribution must be a distrax.Distribution object.")
        if not isinstance(bijector, distrax.Bijector):
            raise ValueError("bijector must be a distrax.Bijector object.")
        
        self.distribution = distribution
        self.bijector = bijector
        if invert_bijector:
            self.bijector = distrax.Inverse(self.bijector)

    def forward(self, x):
        return self.bijector.forward(x)
    def inverse(self, x):
        return self.bijector.inverse(x)
    
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
        x, logdet = self.bijector.inverse_and_log_det(y)
        return self.distribution.log_prob(x) + logdet
    
    def __repr__(self) -> str:
        repr = (
            f"Prior(\n"
            f"  distribution={self.distribution},\n"
            f"  bijector={self.bijector},\n"
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

    #TODO: uncomment when/if neccessary
    # def sample_prior(self, key):
    #     return jax.tree_util.tree_map(lambda dist, subkey: dist.sample(seed=subkey),
    #                                   self.priors,
    #                                   jax.random.split(key, jax.tree_util.tree_leaves(self.priors)))

    #TODO: uncomment when/if neccessary
    # def unconstrain(self, constrained_params_flat):
    #     """
    #     Transform constrained parameters to unconstrained space.
    #     Args:
    #         constrained_params_flat: A flat array of constrained parameters.
    #     Returns:
    #         A flat array of unconstrained parameters.
    #     """
    #     pass
    
    def constrain_sample(self, samples_flat) -> List[Float]:
        """
        Transform samples to the constrained space.
        Args:
            samples_flat: A flat array of samples.
        Returns:
            A tree of samples in the constrained space.
        """
        # Constrain the samples to the constrained space
        return jax.tree.map(
            lambda param_prior, x: param_prior.inverse(x),
            self.priors_flat,
            samples_flat
        )

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
        def log_prior_func(constrained_params_flat: List) -> Float:
            log_probs = jax.tree.map(
                lambda log_prob_func, x: log_prob_func(x),
                self.prior_log_prob_funcs,
                constrained_params_flat
            )
            return jnp.sum(jnp.concatenate([jnp.atleast_1d(x) for x in log_probs]))
        
        return log_prior_func
    
    # def get_kernel_parameters(self, ):
    #     pass

    # def get_unconstrained_shape(self):
    #     leaves = jax.tree_util.tree_leaves(self.priors)
    #     total_params = 0
    #     for leaf in leaves:
    #         # Get the shape of a sample from the distribution
    #         sample_shape = jnp.shape(leaf.sample(seed=jax.random.PRNGKey(0)))
    #         total_params += jnp.prod(jnp.array(sample_shape))
    #     return int(total_params)
    

def _check_prior_dict(params_prior_dict: PriorDict):
    """
    Check if the kernel config is valid.
    """
    # Check if the kernel config has the correct keys
    #TODO: make 'epsilon_eta' optional
    if not set(['thetas', 'eta', 'delta', 'epsilon', 'epsilon_eta']).issubset(list(params_prior_dict.keys())):
        raise ValueError("params_prior_dict keys must contain ['thetas', 'eta', 'delta', 'epsilon', 'epsilon_eta']")
    
    # Check if each kernel has the required keys
    for key, param_prior_dict in params_prior_dict.items():
        if key != 'thetas':
            if 'variance' not in param_prior_dict:
                raise KeyError(f"params_prior_dict key '{key}' must contain 'variance'")
            
            if not isinstance(param_prior_dict['variance'], ParameterPrior):
                raise ValueError(f"params_prior_dict['{key}']['variance'] must be a ParameterPrior instance.")
        
            for param_type in ['lengthscale', 'period']: # Add other types as needed
                if param_type in param_prior_dict:
                    if not isinstance(param_prior_dict[param_type], dict):
                        raise ValueError(f"params_prior_dict['{key}']['{param_type}'] must be a dictionary.")
                    
                    for sub_param_name, sub_param_item in param_prior_dict[param_type].items():
                        if not isinstance(sub_param_item, ParameterPrior):
                            raise ValueError(f"params_prior_dict['{key}']['{param_type}']['{sub_param_name}'] must be a ParameterPrior instance.")
        
        if key == 'thetas':
            if not isinstance(param_prior_dict, dict):
                raise ValueError(f"params_prior_dict['{key}'] must be a dictionary.")
            for sub_param_name, sub_param_item in param_prior_dict.items():
                if not isinstance(sub_param_item, ParameterPrior):
                    raise ValueError(f"params_prior_dict['thetas']['{sub_param_name}'] must be a ParameterPrior instance.")

        #TODO: add checks for parameters depending on the AbstractKernel class.
        # e.g. if kernel_item['kernel'] is RBF, check that lengthscale is a distrax.Distribution
        # How to get the parameters from the kernel class?
        # Is this worth the faff?
