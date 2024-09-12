# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass

import cola
import jax.numpy as jnp
from jaxtyping import (
    Num,
)

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import (
    Gaussian,
)
from gpjax.typing import (
    Array,
)

from kohgpjax.kernel import KOHKernel
from gpjax.gps import *


@dataclass
class KOHPosterior(AbstractPosterior):
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
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        print("Use predict_eta() or predict_obs() methods instead.")
        raise NotImplementedError

    def predict_eta(
        self,
        test_inputs: Num[Array, "N D"],
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

        # Unpack test inputs
        t = test_inputs
        # n_pred = t.shape[0]

        # Observation noise o²
        # obs_noise = self.likelihood.obs_stddev**2 # No longer used as already implemented into kernel
        mx = self.prior.mean_function(x)

        ###### NEW METHOD ######
        # stack the regression and prediction inputs
        x_stack = jnp.vstack((x, t))

        # compute the cross-covariance matrix
        K = self.prior.kernel.cross_covariance(x_stack, x_stack)
        Kxx = cola.PSD(
            cola.ops.Dense(
                K[:n_train, :n_train]
            )
        )
        Kxt = K[:n_train, n_train:]
        Ktt = cola.PSD(
            cola.ops.Dense(
                K[n_train:, n_train:]
            )
        )

        Kxx += cola.ops.I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.prior.jitter
        covariance = cola.PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)


    def predict_obs(
        self,
        test_inputs: Num[Array, "N D"],
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
        test_inputs: Num[Array, "N D"],
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

        # Unpack test inputs
        t = test_inputs
        # n_pred = t.shape[0]

        # Observation noise o²
        # obs_noise = self.likelihood.obs_stddev**2 # No longer used as already implemented into kernel
        mx = self.prior.mean_function(x)

        # Calculate bias terms for prediction
        num_field_obs = self.prior.kernel.num_field_obs
        Kddpred = self.prior.kernel.k_delta.cross_covariance(x[:num_field_obs, :], t)
        Kdpreddpred = self.prior.kernel.k_delta.cross_covariance(t, t)

        # stack the regression and prediction inputs
        x_stack = jnp.vstack((x, t))

        # compute the cross-covariance matrix
        K = self.prior.kernel.cross_covariance(x_stack, x_stack)

        Kxx = cola.PSD(
            cola.ops.Dense(
                K[:n_train, :n_train]
            )
        )
        Kxt = K[:n_train, n_train:] + jnp.pad(Kddpred, ((0, x.shape[0]-num_field_obs), (0, 0)))
        Ktt = cola.PSD(
            cola.ops.Dense(
                K[n_train:, n_train:] + Kdpreddpred
            )
        )
        if include_observation_noise:
            Ktt += self.prior.kernel.k_epsilon.cross_covariance(t, t)

        Kxx += cola.ops.I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.prior.jitter
        covariance = cola.PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)



#######################
# Utils
#######################

def construct_posterior(prior, likelihood):
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



__all__ = [
    "KOHPosterior",
    "construct_posterior"
]