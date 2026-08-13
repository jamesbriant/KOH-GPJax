"""Numerical regression tests for the worked example in ``examples/basics.py``."""

from pathlib import Path

import gpjax as gpx
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import grad, jit, numpy as jnp, tree

from kohgpjax.dataset import KOHDataset
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict, ModelParameters, ParameterPrior


class ExampleModel(KOHModel):
    """The model used by the basics example, kept local as a test oracle."""

    def k_eta(self, params_constrained):
        params = params_constrained["eta"]
        return gpx.kernels.ProductKernel(
            kernels=[
                gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                    variance=jnp.array(1 / params["variances"]["precision"]),
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

    def k_epsilon(self, params_constrained):
        params = params_constrained["epsilon"]
        return gpx.kernels.White(
            active_dims=[0],
            variance=jnp.array(1 / params["variances"]["precision"]),
        )


@pytest.fixture(scope="module")
def example_prior_dict() -> ModelParameterPriorDict:
    return {
        "thetas": {
            "theta_0": ParameterPrior(dist.Uniform(low=0.3, high=0.5), name="theta_0")
        },
        "eta": {
            "variances": {
                "precision": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=4.0),
                    name="eta_precision",
                )
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
                "precision": ParameterPrior(
                    dist.Gamma(concentration=2.0, rate=0.1),
                    name="delta_precision",
                )
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    dist.Gamma(concentration=4.0, rate=2.0),
                    name="delta_lengthscale_x_0",
                )
            },
        },
        "epsilon": {
            "variances": {
                "precision": ParameterPrior(
                    dist.Gamma(concentration=12.0, rate=0.025),
                    name="epsilon_precision",
                )
            }
        },
    }


@pytest.fixture(scope="module")
def example_model(example_prior_dict):
    data_dir = Path(__file__).parents[2] / "examples" / "data"
    field = np.loadtxt(data_dir / "field.csv", delimiter=",", dtype=np.float32)
    simulation = np.loadtxt(data_dir / "sim.csv", delimiter=",", dtype=np.float32)

    field_dataset = gpx.Dataset(
        jnp.asarray(field[:, :1], dtype=jnp.float64),
        jnp.asarray(field[:, 1:], dtype=jnp.float64),
    )
    simulation_dataset = gpx.Dataset(
        jnp.asarray(simulation[:, :2], dtype=jnp.float64),
        jnp.asarray(simulation[:, 2:], dtype=jnp.float64),
    )
    dataset = KOHDataset(field_dataset, simulation_dataset)
    parameters = ModelParameters(example_prior_dict)
    return ExampleModel(kohdataset=dataset, model_parameters=parameters)


def _example_prior_mean_state(example_prior_dict):
    leaves, _ = tree.flatten(example_prior_dict)
    return jnp.asarray(
        [prior.inverse(prior.distribution.mean) for prior in leaves],
        dtype=jnp.float64,
    )


def test_example_posterior_density_has_stable_value_and_gradient(
    example_model, example_prior_dict
):
    """The documented example has a stable eager/JIT numerical contract."""
    state = _example_prior_mean_state(example_prior_dict)
    density = example_model.get_KOH_neg_log_pos_dens_func()

    value = density(state)
    value_jit = jit(density)(state)
    gradient = grad(density)(state)
    gradient_jit = jit(grad(density))(state)

    print("example state:", state)
    print("example density:", value)
    print("example gradient:", gradient)

    expected_value = jnp.array(18302.367801775476, dtype=jnp.float64)
    expected_gradient = jnp.array(
        [
            -7.15381155,
            -3.35597841,
            -5.39850702,
            8.74188395e04,
            1.09173526e05,
            1.12665347e04,
            6.12961004e-01,
        ],
        dtype=jnp.float64,
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(value, expected_value, rtol=1e-8, atol=1e-8)
    assert jnp.allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)
    assert jnp.allclose(value, value_jit, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(gradient, gradient_jit, rtol=1e-6, atol=1e-6)
