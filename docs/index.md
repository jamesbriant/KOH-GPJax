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

=== "Python"
    ```py
    import gpjax as gpx

    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(mean_function = mean, kernel = kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

    posterior = prior * likelihood
    ```

=== "Math"
    $$\begin{align}
    k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
    p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
    p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\ \\
    p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
    \end{align}$$

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
