# KOH-GPJax

## Introduction

KOH-GPJax is an extension to the [GPJax](https://github.com/JaxGaussianProcesses/GPJax) Python package that implements the [Kennedy & O'Hagan (2001)](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00294)[^1] Bayesian Calibration of Computer Models framework.

By combining the power of [Jax](https://jax.readthedocs.io/en/latest/) with the modular design of GPJax, KOH-GPJax provides a Bayesian calibration framework for large-scale computer simulations.

This package is a work in progress. Please get in touch if you're interested in contributing or using the package.

## Installation

Currently only available on GitHub.

```bash
pip install git+https://github.com/jamesbriant/KOH-GPJax.git
```

## Example Usage

Below is an example of how to use the KOH-GPJax framework to perform Bayesian calibration of a computer model.

### Step 1: Define the Model

The model is defined by inheriting from `KOHModel` and implementing the required kernel methods. The `k_epsilon_eta` method is now optional; if omitted, the model will use a zero-variance white noise kernel by default.

```python
# filepath: /model.py
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

### Step 2: Define the Priors

Define the prior distributions and bijectors for all model parameters. The `epsilon_eta` entry is now optional and can be omitted if not needed.

```python
# filepath: /priors.py
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

### Step 3: Load the Data

Load the field and simulation data into `KOHDataset`:

```python
# filepath: /script.py
from jax import config
config.update("jax_enable_x64", True)

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

Note: `sim_dataset.X` must have at least one additional column over `field_dataset.X`.

### Step 4: Initialize the Model and Compute NLPD

Create an instance of the model with the priors and dataset, and compute the negative log posterior density (NLPD):

```python
# filepath: /script.py
from kohgpjax.parameters import ModelParameters
from jax import jit, grad

from priors import prior_dict
from model import Model

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

## Development Setup with Hatch

This project uses [Hatch](https://hatch.pypa.io/latest/) for dependency management and running development tasks, with [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting.

1. **Install Hatch:**
    If you don't have Hatch installed, you can install it via pip:
  
    ```bash
    pip install hatch
    ```

2. **Set up the environment:**
    Navigate to the project root directory and create the development environment:

    ```bash
    hatch env create
    ```

    This will install all project dependencies and development tools defined in `pyproject.toml`.

3. **Activate the environment:**
    To activate the Hatch-managed environment, run:

    ```bash
    hatch shell
    ```

    You are now in a shell with all dependencies available.

4. **Running tasks:**
    Common development tasks are defined as scripts in `pyproject.toml` and can be run using `hatch run <env>:<script_name>`. For the default development environment (`dev`):

    * **Run tests:**

        ```bash
        hatch run dev:test
        ```

    * **Check linting and formatting:**

        ```bash
        hatch run dev:check
        ```

        This runs `ruff check --fix` to check and automatically fix linting issues, import sorting, and remove unused variables.

    * **Apply formatting:**

        ```bash
        hatch run dev:format
        ```

        This runs `ruff format` on the codebase and formats Jupyter notebooks using `jupytext`.

    * **Run all checks and tests:**

        ```bash
        hatch run dev:all-tests
        ```

    * **View test coverage report:**
        First, generate the coverage data:

        ```bash
        hatch run dev:coverage
        ```

        Then, you can open `htmlcov/index.html` in your browser, or view the XML report in `coverage.xml`.

    * **Check docstrings:**

        ```bash
        hatch run dev:docstrings
        ```

Refer to the `[tool.hatch.envs.dev.scripts]` section in `pyproject.toml` for all available scripts.

## References

[^1]: Kennedy, M.C. and O'Hagan, A. (2001), Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 425-464. [https://doi.org/10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294)
