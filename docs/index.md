# Welcome to KOH-GPJax

KOH-GPJax is a modern Python implementation of the Bayesian Calibration
of Computer Models framework first outlined by [Kennedy & O'Hagan (2001) [external]](https://doi.org/10.1111/1467-9868.00294)[^1].
This package inherits the GPU acceleration and just-in-time compilation
features of [JAX [external]](https://docs.jax.dev/en/latest/index.html) and builds upon the flexible yet elegant Gaussian processes package
[GPJax [external]](https://docs.jaxgaussianprocesses.com).

## Basic Example

Bayesian calibration procedures require three ingredients: Gaussian process kernel defintions,
parameter prior definitions and a posterior sampler. KOH-GPJax provides the infrastucture
to build the Bayesian model using the first two components provided by the user and
exposes a log posterior density function for the MCMC sampler of your choosing.

The data for this problem can be found in `examples/data/`.

=== "`main.py`"
    ```py
    import jax.numpy as jnp
    from jax import config, grad, jit
    from kohgpjax.parameters import ModelParameters

    from dataloader import kohdataset
    from model import Model
    from priors import prior_dict

    config.update("jax_enable_x64", True)

    model_parameters = ModelParameters(prior_dict=prior_dict)

    model = Model(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )
    nlpd_func = model.get_KOH_neg_log_pos_dens_func()

    # JIT-compile the NLPD function
    nlpd_jitted = jit(nlpd_func)

    # Compute the gradient of the NLPD
    grad_nlpd_jitted = jit(grad(nlpd_func))

    # Example usage
    # Alternatively take the mean of each parameter's prior distribution.
    example_params = jnp.array([0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nlpd_value = nlpd_jitted(example_params)
    nlpd_gradient = grad_nlpd_jitted(example_params)

    print("NLPD Value:", nlpd_value)
    print("NLPD Gradient:", nlpd_gradient)
    ```

=== "`model.py`"
    ```py
    import gpjax as gpx
    import jax.numpy as jnp
    from kohgpjax.kohmodel import KOHModel

    class Model(KOHModel):
        def k_eta(self, params_constrained):
            params = params_constrained["eta"]
            return gpx.kernels.ProductKernel(
                kernels=[
                    gpx.kernels.RBF(
                        active_dims=[0],
                        lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                        variance=jnp.array(
                            1
                            / params["variances"][
                                "precision"
                            ]  # Precision consistent with Higdon et al. (2004)
                        ),
                    ),
                    gpx.kernels.RBF(
                        active_dims=[1],
                        lengthscale=jnp.array(params["lengthscales"]["theta_0"]),
                    ),
                ]
            )

        def k_delta(self, params_constrained):
            params = params_constrained["delta"]
            return gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                variance=jnp.array(1 / params["variances"]["precision"]),
            )

        # Definition of k_epsilon is optional in model. Defaults behaviour is to use a White kernel.
        # Alternative observation noise kernels should be defined here if needed.
        # The prior for ["epsilon"]["variances"]["variance"] is still required in priors.py
        # even if user defined k_epsilon is not provided.
        def k_epsilon(self, params_constrained):
            params = params_constrained["epsilon"]
            return gpx.kernels.White(
                #
                active_dims=[0],
                variance=jnp.array(1 / params["variances"]["precision"]),
            )

        # k_epsilon_eta is completely optional. Default behaviour is a white kernel with
        # variance=0 effectively turning off this component.
        # def k_epsilon_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        #     params = params_constrained['epsilon_eta']
        #     return gpx.kernels.White(
        #         active_dims=[0],
        #         variance=jnp.array(1/params['variances']['precision'])
        #     )
    ```

=== "`priors.py`"
    ```py
    import numpyro.distributions as dist
    from kohgpjax.parameters import ModelParameterPriorDict, ParameterPrior

    prior_dict: ModelParameterPriorDict = {
        "thetas": {
            "theta_0": ParameterPrior(
                dist.Uniform(low=0.3, high=0.5),
                name="theta_0",
            ),
        },
        "eta": {
            "variances": {
                "precision": ParameterPrior(  # Precision consistent with Higdon et al. (2004)
                    dist.Gamma(concentration=2.0, rate=4.0),
                    name="eta_precision",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    dist.Gamma(concentration=4.0, rate=1.4),
                    name="eta_lengthscale_x_0",
                ),
                "theta_0": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=3.5),
                    name="eta_lengthscale_theta_0",
                ),
            },
        },
        "delta": {
            "variances": {
                "precision": ParameterPrior(  # Precision consistent with Higdon et al. (2004)
                    dist.Gamma(concentration=2.0, rate=0.1),
                    name="delta_precision",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    dist.Gamma(
                        concentration=5.0, rate=0.3
                    ),  # encourage long value => linear discrepancy
                    name="delta_lengthscale_x_0",
                ),
            },
        },
        "epsilon": {
            "variances": {
                "precision": ParameterPrior(  # Precision consistent with Higdon et al. (2004)
                    dist.Gamma(concentration=800, rate=2.0),
                    name="epsilon_precision",
                ),
            },
        },
    }
    ```

=== "`dataloader.py`"
    ```py
    import gpjax as gpx
    import jax.numpy as jnp
    import numpy as np
    from jax import config
    from kohgpjax.dataset import KOHDataset

    config.update("jax_enable_x64", True)

    DATAFIELD = np.loadtxt("field.csv", delimiter=",", dtype=np.float32)
    DATASIM = np.loadtxt("sim.csv", delimiter=",", dtype=np.float32)

    xf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATASIM[:, 0], (-1, 1)).astype(jnp.float64)
    tc = jnp.reshape(DATASIM[:, 1], (-1, 1)).astype(jnp.float64)
    yf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    yc = jnp.reshape(DATASIM[:, 2], (-1, 1)).astype(jnp.float64)

    field_dataset = gpx.Dataset(xf, yf)
    sim_dataset = gpx.Dataset(jnp.hstack((xc, tc)), yc)

    kohdataset = KOHDataset(field_dataset, sim_dataset)
    ```

<!-- === "Math"
    $$\begin{align}
    k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
    p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
    p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\ \\
    p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
    \end{align}$$ -->

<!-- ## Quick start

!!! Install

    GPJax can be installed via pip. See our [installation guide](installation.md) for further details.

    ```bash
    pip install gpjax
    ```

!!! New

    New to GPs? Then why not check out our [introductory notebook](_examples/intro_to_gps.md) that starts from Bayes' theorem and univariate Gaussian distributions.

!!! Begin

    Looking for a good place to start? Then why not begin with our [regression
    notebook](https://docs.jaxgaussianprocesses.com/_examples/regression/). -->

[^1]: Kennedy, M.C. and O'Hagan, A. (2001), Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 425-464. [https://doi.org/10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294)
