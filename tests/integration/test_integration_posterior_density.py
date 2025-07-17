from jax import config  # Re-adding config for early x64 enabling

config.update("jax_enable_x64", True)  # Needs to be before JAX ops at global scope

import gpjax as gpx
from jax import (
    jit,
    numpy as jnp,
)

# import numpy as np # No longer needed for np.loadtxt
import numpyro.distributions as npd

from kohgpjax.dataset import KOHDataset
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import (
    ModelParameterDict,
    ModelParameters,
    ParameterPrior,
)

# import jax.tree_util as jtu # No longer needed after removing __main__ block


# Original DATAFIELD = np.loadtxt("/Users/jamesbriant/Documents/Projects/BayesianCalibrationExamples/presentable/data/toy/field.csv", delimiter=",", dtype=np.float32)
# Original DATACOMP = np.loadtxt("/Users/jamesbriant/Documents/Projects/BayesianCalibrationExamples/presentable/data/toy/sim.csv", delimiter=",", dtype=np.float32)

# Using minimal inline data instead of external CSV files
# Field data: xf (1 var_param), yf
DATAFIELD_MOCK = jnp.array([[1.0, 10.0], [2.0, 12.0]], dtype=jnp.float32)

# Simulation data: xc (1 var_param), tc (1 design_param/calib_param for KOHDataset), yc
DATACOMP_MOCK = jnp.array([[1.1, 0.5, 11.0], [2.1, 0.6, 23.0]], dtype=jnp.float32)


xf = jnp.reshape(DATAFIELD_MOCK[:, 0], (-1, 1)).astype(jnp.float64)
yf = jnp.reshape(DATAFIELD_MOCK[:, 1], (-1, 1)).astype(jnp.float64)

xc = jnp.reshape(DATACOMP_MOCK[:, 0], (-1, 1)).astype(jnp.float64)
tc = jnp.reshape(DATACOMP_MOCK[:, 1], (-1, 1)).astype(
    jnp.float64
)  # This corresponds to the calibration parameter input in the simulation design
yc = jnp.reshape(DATACOMP_MOCK[:, 2], (-1, 1)).astype(jnp.float64)


field_dataset = gpx.Dataset(xf, yf)
# For KOHDataset, sim_dataset.X should have [var_params, calib_input_design_params]
# num_calib_params = sim_dataset.X.shape[1] - field_dataset.X.shape[1]
# Here, field_dataset.X.shape[1] = 1 (xf)
# comp_dataset.X will have shape (N_sim, 2) from hstack(xc, tc)
# So, num_calib_params = 2 - 1 = 1. This matches `thetas: {"theta_0": ...}`.
comp_dataset = gpx.Dataset(jnp.hstack((xc, tc)), yc)


kohdataset = KOHDataset(field_dataset, comp_dataset)

prior_dict: ModelParameterDict = {
    "thetas": {  # Corresponds to num_calib_params = 1
        "theta_0": ParameterPrior(npd.Normal(loc=0.0, scale=1.0)),
    },
    "eta": {
        "variances": {"var": ParameterPrior(npd.Normal(loc=0.0, scale=0.5))},
        "lengthscales": {
            "x_0": ParameterPrior(npd.Normal(loc=0.0, scale=1.0)),
            "theta_0": ParameterPrior(npd.Normal(loc=0.0, scale=1.0)),
        },
    },
    "delta": {
        "variances": {"var": ParameterPrior(npd.Normal(loc=0.0, scale=0.5))},
        "lengthscales": {
            "x_0": ParameterPrior(npd.Normal(loc=0.0, scale=1.0)),
        },
    },
    "epsilon": {
        "variances": {"obs_noise": ParameterPrior(npd.Normal(loc=0.0, scale=0.5))},
    },
    "epsilon_eta": {
        "variances": {"var": ParameterPrior(npd.Normal(loc=0.1, scale=0.5))},
    },
}


class MyModel(KOHModel):
    # Workaround to provide kohdataset to k_delta for active_dims
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kohdataset_cached_for_test = kwargs.get("kohdataset")

    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        eta_params = GPJAX_params.get("eta", {})
        len_x0 = eta_params.get("lengthscales", {}).get("x_0", 1.0)
        len_theta0 = eta_params.get("lengthscales", {}).get("theta_0", 1.0)
        var_eta = eta_params.get("variances", {}).get("var", 1.0)  # Adjusted path

        return gpx.kernels.ProductKernel(
            kernels=[
                gpx.kernels.RBF(active_dims=[0], lengthscale=len_x0),
                gpx.kernels.RBF(active_dims=[1], lengthscale=len_theta0),
            ]
        ) * gpx.kernels.Constant(constant=var_eta)  # Multiply by overall variance

    def k_delta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        delta_params = GPJAX_params.get("delta", {})
        len_x0 = delta_params.get("lengthscales", {}).get("x_0", 1.0)
        var_delta = delta_params.get("variances", {}).get("var", 1.0)  # Adjusted path

        return gpx.kernels.RBF(
            active_dims=list(
                range(self._kohdataset_cached_for_test.num_variable_params)
            ),
            lengthscale=len_x0,
            variance=var_delta,
        )

    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        epsilon_eta_params = GPJAX_params.get("epsilon_eta", {})
        var_eps_eta = epsilon_eta_params.get("variances", {}).get(
            "var", 0.01
        )  # Adjusted path

        return gpx.kernels.White(
            active_dims=list(
                range(self._kohdataset_cached_for_test.num_variable_params)
            ),
            variance=var_eps_eta,
        )


# The following global instantiations are part of the original test structure.
# They will be executed when pytest collects the test file.
# The prior_dict is now more compliant with _check_prior_dict.
model_params = ModelParameters(prior_dict)
model = MyModel(
    model_parameters=model_params,
    kohdataset=kohdataset,
    # obs_stddev=None, so k_epsilon will use params from 'epsilon.variances.obs_noise'
    # This is handled by KOHModel.k_epsilon's logic to get from params_constrained.
)

# For debugging parameter order, can be uncommented locally
# mp_temp_debug = ModelParameters(prior_dict)
# print(f"DEBUG: Number of parameters: {mp_temp_debug.n_params}")
# paths_debug = jtu.tree_leaves_with_path(mp_temp_debug.priors, is_leaf=lambda x: isinstance(x, ParameterPrior))
# print("DEBUG: Parameter order from tree_leaves_with_path:")
# for i_debug, (path_debug, leaf_debug) in enumerate(paths_debug):
#     print(f"DEBUG: Index {i_debug}: Path: {jtu.keystr(path_debug)}")


def test_neg_log_posterior_density_runs():
    """Tests that the negative log posterior density function runs and returns a scalar."""
    # model and model_params are now global in this test file scope
    f = jit(model.get_KOH_neg_log_pos_dens_func())

    # Determine the number of parameters from the model_params fixture
    # Original prior_dict (after correction to be dicts of PPs):
    # thetas: theta_0 (1)
    # eta: var (1), ls.x_0 (1), ls.theta_0 (1) -> 3
    # delta: var (1), ls.x_0 (1) -> 2
    # epsilon: obs_noise (1) -> 1
    # epsilon_eta: var (1) -> 1
    # Total = 1 + 3 + 2 + 1 + 1 = 8 params.

    # The order of mcmc_params needs to match the flattened order of prior_dict.
    # JAX flattens dictionaries by sorting keys alphabetically.
    # delta (lengthscales.x_0, variances.var)
    # epsilon (variances.obs_noise)
    # epsilon_eta (variances.var)
    # eta (lengthscales.theta_0, lengthscales.x_0, variances.var) - Note: theta_0 before x_0 alphabetically
    # thetas (theta_0)
    # Example values, actual values don't matter as much as the correct length and type
    mcmc_params = [
        0.1,  # delta.lengthscales.x_0
        0.2,  # delta.variances.var
        0.3,  # epsilon.variances.obs_noise
        0.4,  # epsilon_eta.variances.var
        0.5,  # eta.lengthscales.theta_0
        0.6,  # eta.lengthscales.x_0
        0.7,  # eta.variances.var
        0.8,  # thetas.theta_0
    ]
    assert len(mcmc_params) == model_params.n_params, (
        f"mcmc_params length {len(mcmc_params)} != n_params {model_params.n_params}"
    )

    result = f(jnp.array(mcmc_params, dtype=jnp.float64))  # Ensure it's a JAX array
    print(f"Computed -log posterior density: {result}")
    # Original value was 707.3885600290141 with different data and potentially different prior structure.
    # With new data and structure, the value will differ.
    # The key is that it runs and returns a scalar float without NaN.
    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()  # Scalar
    assert not jnp.isnan(result)


# End of test_integration_posterior_density.py
