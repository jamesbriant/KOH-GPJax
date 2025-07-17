import beartype.typing as tp
from cola.annotations import PSD
from cola.linalg.inverse.inv import solve
from cola.ops import Dense
from cola.ops.operators import I_like
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    AbstractPosterior,
    AbstractPrior,
    ConjugatePosterior,
    NonConjugatePosterior,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.typing import Array
import jax.numpy as jnp
from jaxtyping import Num

from kohgpjax.kernels.kohkernel import KOHKernel

K = tp.TypeVar("K", bound=AbstractKernel)  # noqa: F821
M = tp.TypeVar("M", bound=AbstractMeanFunction)  # noqa: F821
P = tp.TypeVar("P", bound=AbstractPrior)  # noqa: F821
PKOH = tp.TypeVar("PKOH", bound=KOHKernel)
L = tp.TypeVar("L", bound=AbstractLikelihood)
NGL = tp.TypeVar("NGL", bound=NonGaussian)  # noqa: F821
GL = tp.TypeVar("GL", bound=Gaussian)  # noqa: F821


class KOHPosterior(AbstractPosterior[PKOH, GL]):
    r"""A Conjuate Gaussian process posterior object.

    A Gaussian process posterior distribution when the constituent likelihood
    function is a Gaussian distribution. In such cases, the latent function values
    $`f`$ can be analytically integrated out of the posterior distribution.
    As such, many computational operations can be simplified; something we make use
    of in this object.

    For a Gaussian process prior $`p(\mathbf{f})`$ and a Gaussian likelihood
    $`p(y | \mathbf{f}) = \mathcal{N}(y\mid \mathbf{f}, \sigma^2))`$ where
    $`\mathbf{f} = f(\mathbf{x})`$, the predictive posterior distribution at
    a set of inputs $`\mathbf{x}`$ is given by
    ```math
    \begin{align}
    p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star}, \mathbf{f} \mid \mathbf{y})\\
        & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
    \end{align}
    ```
    where
    ```math
    \begin{align}
    \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
    \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
    \end{align}
    ```

    Example:
        ```python
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp

            >>> prior = gpx.gps.Prior(
                    mean_function = gpx.mean_functions.Zero(),
                    kernel = gpx.kernels.RBF()
                )
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
            >>>
            >>> posterior = prior * likelihood
        ```
    """

    def predict(
        self,
        test_inputs,  #: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        raise NotImplementedError(
            "Use predict_eta(), predict_zeta() or predict_obs() methods instead."
        )

    def predict_eta(
        self,
        test_inputs,  #: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.gps.Prior(mean_function = gpx.mean_functions.Zero(), kernel = gpx.kernels.RBF())
                >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A function that accepts an input array and
                returns the predictive distribution as a `GaussianDistribution`.
        """
        # Unpack training data
        x, y = train_data.X, train_data.y
        n_train = x.shape[0]
        n_obs = self.prior.kernel.num_field_obs

        # Unpack test inputs
        t = test_inputs

        # Observation noise o²
        obs_var = self.likelihood.obs_stddev.value**2
        mx = self.prior.mean_function(x)

        ###### NEW METHOD ######
        # stack the regression and prediction inputs
        x_stack = jnp.vstack((x, t))

        # compute the cross-covariance matrix
        K = self.prior.kernel.cross_covariance(
            x_stack, x_stack
        )  # need array, not the cola linear operator so use cross_covariance() method not gram() method
        Kxx = K[:n_train, :n_train]
        Kxt = K[:n_train, n_train:]
        Ktt = PSD(Dense(K[n_train:, n_train:]))

        # Σ = Kxx + Io²
        Kxx += jnp.diag(
            jnp.pad(
                jnp.ones(n_obs) * obs_var,
                (0, x.shape[0] - n_obs),
            )
        )
        Kxx += jnp.identity(Kxx.shape[0]) * self.jitter
        Sigma = PSD(Dense(Kxx))
        Sigma_inv_Kxt = solve(
            Sigma, Kxt
        )  # GPJax 0.9.3 enforces Cholesky algorithm here. I choose to let cola decide the best algorithm.

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean_t = self.prior.mean_function(t)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, #TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += I_like(covariance) * self.prior.jitter
        covariance = PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

    def predict_obs(
        self,
        test_inputs,  #: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.gps.Prior(mean_function = gpx.mean_functions.Zero(), kernel = gpx.kernels.RBF())
                >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A function that accepts an input array and
                returns the predictive distribution as a `GaussianDistribution`.
        """
        return self.predict_zeta(
            test_inputs=test_inputs,
            train_data=train_data,
            include_observation_noise=True,
        )

    def predict_zeta(
        self,
        test_inputs,  #: Num[Array, "N D"],
        train_data: Dataset,
        include_observation_noise: bool = False,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.gps.Prior(mean_function = gpx.mean_functions.Zero(), kernel = gpx.kernels.RBF())
                >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A function that accepts an input array and
                returns the predictive distribution as a `GaussianDistribution`.
        """
        # Unpack training data
        x, y = train_data.X, train_data.y
        n_train = x.shape[0]
        n_obs = self.prior.kernel.num_field_obs

        # Unpack test inputs
        t = test_inputs
        # n_pred = t.shape[0]

        # Observation noise o²
        obs_var = (
            self.likelihood.obs_stddev.value**2
        )  # No longer used as already implemented into kernel
        mx = self.prior.mean_function(x)

        # Calculate bias terms for prediction
        num_field_obs = self.prior.kernel.num_field_obs
        Kddpred = self.prior.kernel.k_delta.cross_covariance(x[:num_field_obs, :], t)
        Kdpreddpred = self.prior.kernel.k_delta.cross_covariance(t, t)

        # stack the regression and prediction inputs
        x_stack = jnp.vstack((x, t))

        # compute the cross-covariance matrix
        K = self.prior.kernel.cross_covariance(x_stack, x_stack)

        Kxx = K[:n_train, :n_train]
        Kxt = K[:n_train, n_train:] + jnp.pad(
            Kddpred, ((0, x.shape[0] - num_field_obs), (0, 0))
        )
        Ktt = K[n_train:, n_train:] + Kdpreddpred
        Ktt = PSD(Dense(Ktt))

        if (
            include_observation_noise
        ):  # This cannot be jitted. TODO: Find a way to make this jittable.
            Ktt += self.prior.kernel.k_epsilon.cross_covariance(t, t)

        # Kxx += I_like(Kxx) * self.jitter
        # Σ = Kxx + Io²
        Kxx += jnp.diag(
            jnp.pad(
                jnp.ones(n_obs) * obs_var,
                (0, x.shape[0] - n_obs),
            )
        )
        Kxx += jnp.identity(Kxx.shape[0]) * self.jitter
        Sigma = PSD(Dense(Kxx))  # + cola.ops.I_like(Kxx) * obs_var
        # Sigma = PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        Sigma_inv_Kxt = solve(
            Sigma, Kxt
        )  # GPJax 0.9.3 enforces Cholesky algorithm here. I choose to let cola decide the best algorithm.

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += I_like(covariance) * self.prior.jitter
        covariance = PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)


#######################
# Utils
#######################


@tp.overload
def construct_posterior(prior: P, likelihood: GL) -> ConjugatePosterior[P, GL]: ...


@tp.overload
def construct_posterior(  # noqa: F811
    prior: P, likelihood: NGL
) -> NonConjugatePosterior[P, NGL]: ...


@tp.overload
def construct_posterior(  # noqa: F811
    prior: PKOH, likelihood: GL
) -> KOHPosterior[PKOH, GL]: ...


def construct_posterior(prior, likelihood):  # noqa: F811
    r"""Utility function for constructing a posterior object from a prior and
    likelihood. The function will automatically select the correct posterior
    object based on the likelihood.

    Args:
        prior (Prior): The Prior distribution.
        likelihood (AbstractLikelihood): The likelihood that represents our
            beliefs around the distribution of the data.

    Returns
    -------
        AbstractPosterior: A posterior distribution. If the likelihood is
            Gaussian, then a `ConjugatePosterior` will be returned. Otherwise,
            a `NonConjugatePosterior` will be returned.
    """
    if isinstance(likelihood, Gaussian):
        if isinstance(prior.kernel, KOHKernel):
            return KOHPosterior(prior=prior, likelihood=likelihood)

        return ConjugatePosterior(prior=prior, likelihood=likelihood)

    return NonConjugatePosterior(prior=prior, likelihood=likelihood)


__all__ = ["KOHPosterior", "construct_posterior"]
