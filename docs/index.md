# Welcome to KOH-GPJax

KOH-GPJax is a modern Python implementation of the Bayesian Calibration
of Computer Models framework first outlined by Kennedy & O'Hagan (2001)[^1].
This package inherits the GPU acceleration and just-in-time compilation
features of JAX and the flexible yet elegant Gaussian processes package
GPJax.

## Basic Example

Bayesian calibration procedures require three ingredients: Gaussian process kernel defintions,
parameter prior definitions and a posterior sampler. KOH-GPJax provides the infrastucture
to build the Bayesian model using the first two components provided by the user and
exposes a log posterior density function for the MCMC sampler of your choosing.

=== "`script.py`"
    ```py
    from jax import config
    config.update("jax_enable_x64", True)

    from kohgpjax.parameters import ModelParameters
    from jax import jit, grad

    from data import kohdataset
    from model import Model
    from priors import prior_dict

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
    example_params = jnp.array([0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
        def k_eta(self, eta_params_constrained):
            return gpx.kernels.ProductKernel(
                kernels=[
                    gpx.kernels.RBF(
                        active_dims=[0],
                        lengthscale=jnp.array(eta_params_constrained['lengthscales']['x_0']),
                        variance=jnp.array(1 / eta_params_constrained['variances']['variance']),
                    ),
                    gpx.kernels.RBF(
                        active_dims=[1],
                        lengthscale=jnp.array(eta_params_constrained['lengthscales']['theta_0']),
                    ),
                ]
            )

        def k_delta(self, delta_params_constrained):
            return gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(delta_params_constrained['lengthscales']['x_0']),
                variance=jnp.array(1 / delta_params_constrained['variances']['variance']),
            )

        # k_epsilon_eta is now optional. If not defined, a zero-variance white kernel is used.
        # def k_epsilon_eta(self, epsilon_eta_params_constrained=None):
        #     if epsilon_eta_params_constrained is None:
        #         return None
        #     return gpx.kernels.White(
        #         active_dims=[0],
        #         variance=jnp.array(1 / epsilon_eta_params_constrained['variances']['variance']),
        #     )
    ```

=== "`priors.py`"
    ```py
    import distrax
    from kohgpjax.parameters import ParameterPrior, PriorDict

    prior_dict: PriorDict = {
        'thetas': {
            'theta_0': ParameterPrior(
                distrax.Uniform(low=0.3, high=0.5),
                distrax.Chain([
                    distrax.Inverse(distrax.Tanh()),
                    distrax.Lambda(lambda x: 2 * (x - 0.3) / (0.5 - 0.3) - 1),
                ]),
                name='theta_0',
            ),
        },
        'eta': {
            'variances': {
                'variance': ParameterPrior(
                    distrax.Gamma(concentration=2.0, rate=1.0),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='eta_variance',
                ),
            },
            'lengthscales': {
                'x_0': ParameterPrior(
                    distrax.Gamma(concentration=2.0, rate=1.0),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='eta_lengthscale_x_0',
                ),
                'theta_0': ParameterPrior(
                    distrax.Gamma(concentration=2.0, rate=3.5),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='eta_lengthscale_theta_0',
                ),
            },
        },
        'delta': {
            'variances': {
                'variance': ParameterPrior(
                    distrax.Gamma(concentration=10.0, rate=0.33),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='delta_variance',
                ),
            },
            'lengthscales': {
                'x_0': ParameterPrior(
                    distrax.Gamma(concentration=2.0, rate=1.0),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='delta_lengthscale_x_0',
                ),
            },
        },
        'epsilon': {
            'variances': {
                'variance': ParameterPrior(
                    distrax.Gamma(concentration=12.0, rate=0.025),
                    distrax.Lambda(lambda x: jnp.log(x)),
                    name='epsilon_variance',
                ),
            },
        },
        # 'epsilon_eta' is now optional and can be omitted if not needed.
        # 'epsilon_eta': {
        #     'variances': {
        #         'variance': ParameterPrior(
        #             distrax.Gamma(concentration=10.0, rate=0.001),
        #             distrax.Lambda(lambda x: jnp.log(x)),
        #             name='epsilon_eta_variance',
        #         ),
        #     },
        # },
    }
    ```

=== "`data.py`"
    ```py
    import numpy as np
    import jax.numpy as jnp
    import gpjax as gpx
    from kohgpjax.dataset import KOHDataset

    DATAFIELD = np.loadtxt('field.csv', delimiter=',', dtype=np.float32)
    DATASIM = np.loadtxt('sim.csv', delimiter=',', dtype=np.float32)

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
