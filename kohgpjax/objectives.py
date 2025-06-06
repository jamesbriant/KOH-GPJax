from cola.annotations import PSD
from cola.ops import Dense
from cola.ops.operators import I_like
import jax.numpy as jnp

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.typing import ScalarFloat

from kohgpjax.gps import KOHPosterior


def conjugate_mll(posterior: KOHPosterior, data: Dataset) -> ScalarFloat:
    r"""Evaluate the marginal log-likelihood of the Gaussian process where observation
    noise applies only to the field observations, not the simulation outputs.

    Compute the marginal log-likelihood function of the Gaussian process.
    The returned function can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation
    given here enables exact estimation of the Gaussian process' latent
    function values.

    For a training dataset $\{x_n, y_n\}_{n=1}^N$, set of test inputs
    $\mathbf{x}^{\star}$ the corresponding latent function evaluations are given
    by $\mathbf{f}=f(\mathbf{x})$ and $\mathbf{f}^{\star}f(\mathbf{x}^{\star})$,
    the marginal log-likelihood is given by:

    ```math
    \begin{align}
        \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})
        p(\mathbf{f}, \mathbf{f}^{\star})\mathrm{d}\mathbf{f}^{\star}\\
        & = 0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}')
        + \sigma^2\mathbf{I}_N\right)^{-1}\mathbf{y} \right.\\
        & \quad\left. -\log\lvert k(\mathbf{x}, \mathbf{x}')
        + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
    \end{align}
    ```

    Example:
        >>> import gpjax as gpx

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> gpx.objectives.conjugate_mll(posterior, D)

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising
        the model's parameters with respect to the parameters, we use the negative
        marginal log-likelihood. This can be realised through

        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

    Args:
        posterior (KOHPosterior): The posterior distribution for which
            we want to compute the marginal log-likelihood.
        data:: The training dataset used to compute the
            marginal log-likelihood.

    Returns
    -------
        ScalarFloat: The marginal log-likelihood of the Gaussian process.
    """

    x, y = data.X, data.y
    n_obs = posterior.prior.kernel.num_field_obs

    # Observation noise o²
    obs_noise = posterior.likelihood.obs_stddev.value**2
    mx = posterior.prior.mean_function(x)

    ###### NEW METHOD ######
    # compute the cross-covariance matrix
    # Σ = (Kxx + Io²) = LLᵀ
    Kxx = posterior.prior.kernel.cross_covariance(x, x) # need array, not the cola linear operator so use cross_covariance() method not gram() method
    Kxx += jnp.diag(
        jnp.pad(
            jnp.ones(n_obs) * obs_noise,
            (0, x.shape[0]-n_obs),
        )
    )
    Kxx = Dense(Kxx)
    Sigma = Kxx + I_like(Kxx) * posterior.prior.jitter
    Sigma = PSD(Sigma)

    # p(y | x, θ), where θ are the model hyperparameters:
    mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)
    return mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
