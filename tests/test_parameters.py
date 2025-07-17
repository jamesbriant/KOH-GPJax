import jax.tree_util as jtu
import numpyro.distributions as npd
import pytest
from jax import (  # Removed 'config'
    jit,
)
from jax import (
    numpy as jnp,
)
from kohgpjax.parameters import (
    ModelParameterDict,
    ModelParameters,
    ParameterPrior,
)

# --- Fixtures ---

# The fixture 'sample_prior_dict_corrected' was unused. Removed.

# Comments on _check_prior_dict structure are useful, keeping them.
# _check_prior_dict structure for non-"thetas" keys:
# prior_dict[key]['variances'] (dict of ParameterPrior)
# prior_dict[key]['lengthscales'] (dict of ParameterPrior)
# prior_dict[key]['biases'] (dict of ParameterPrior)
# prior_dict[key]['means'] (dict of ParameterPrior)


@pytest.fixture(scope="module")
def valid_prior_dict() -> ModelParameterDict:
    return {
        "thetas": {  # This is a dict of ParameterPriors
            "calib_param1": ParameterPrior(npd.Normal(0, 1)),  # param_idx: 0
            "calib_param2": ParameterPrior(npd.Normal(0, 1)),  # param_idx: 1
        },
        "eta": {  # This should contain 'variances', 'lengthscales', etc.
            "variances": {
                "kernel_var": ParameterPrior(npd.Normal(1, 0.5))
            },  # param_idx: 2
            "lengthscales": {
                "ls_x": ParameterPrior(npd.Normal(-1, 1)),  # param_idx: 3
                "ls_theta": ParameterPrior(npd.Normal(0.5, 0.25)),  # param_idx: 4
            },
        },
        "delta": {  # Similar structure to eta
            "variances": {
                "kernel_var": ParameterPrior(npd.Normal(0, 1))
            },  # param_idx: 5
            "lengthscales": {"ls_x": ParameterPrior(npd.Normal(0, 1))},  # param_idx: 6
        },
        "epsilon": {  # Optional, not strictly checked by top-level of _check_prior_dict beyond presence
            # For internal structure, it would fall into the 'else' of _check_prior_dict
            # which expects 'variances', 'lengthscales' etc.
            "variances": {
                "obs_noise": ParameterPrior(npd.Normal(-0.5, 0.1))
            }  # param_idx: 7
        },
        # epsilon_eta is also optional. Let's omit for simplicity for now, n_params = 8
    }


@pytest.fixture(scope="module")
def model_params(valid_prior_dict: ModelParameterDict) -> ModelParameters:
    return ModelParameters(valid_prior_dict)


@pytest.fixture(scope="module")
def mcmc_sample_unconstrained(model_params: ModelParameters) -> list[float]:
    # Must match the number of parameters in valid_prior_dict (8 params)
    # Order is determined by jax.tree.flatten on valid_prior_dict
    # JAX flattens dicts by sorting keys alphabetically at each level.
    # delta (variances/kernel_var:0, lengthscales/ls_x:1)
    # epsilon (variances/obs_noise:2)
    # eta (variances/kernel_var:3, lengthscales/ls_x:4, lengthscales/ls_theta:5)
    # thetas (calib_param1:6, calib_param2:7)
    # Let's re-verify this order once model_params.priors_flat can be inspected or inferred.
    # For now, assume this order for expected values:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 8 values


# --- Helper Functions ---


def assert_trees_equal(tree1, tree2, rtol=1e-7, atol=1e-9):
    flat1, treedef1 = jtu.tree_flatten(tree1)
    flat2, treedef2 = jtu.tree_flatten(tree2)
    assert treedef1 == treedef2, (
        f"Tree structures do not match: {treedef1} vs {treedef2}"
    )
    for i, (v1, v2) in enumerate(zip(flat1, flat2, strict=False)):
        assert jnp.allclose(v1, v2, rtol=rtol, atol=atol), (
            f"Leaf {i} mismatch: {v1} vs {v2}"
        )


# --- Tests ---


def test_model_parameters_initialization(
    model_params: ModelParameters, valid_prior_dict: ModelParameterDict
):
    assert model_params.priors == valid_prior_dict
    assert model_params.n_params == 8

    for prior in model_params.priors_flat:
        assert isinstance(prior, ParameterPrior)
        assert prior.bijector.__class__.__name__ == "IdentityTransform"


def test_get_log_prior_func(
    model_params: ModelParameters, mcmc_sample_unconstrained: list[float]
):
    log_prior_func = model_params.get_log_prior_func()
    jitted_log_prior_func = jit(log_prior_func)

    # Expected log prior calculation based on valid_prior_dict and assumed mcmc_sample order
    # JAX sorts dictionary keys for flattening: delta, epsilon, eta, thetas
    # delta:
    #   lengthscales/ls_x: Normal(0,1) -> val: mcmc_sample_unconstrained[0] = 0.1
    #   variances/kernel_var: Normal(0,1) -> val: mcmc_sample_unconstrained[1] = 0.2
    # epsilon:
    #   variances/obs_noise: Normal(-0.5, 0.1) -> val: mcmc_sample_unconstrained[2] = 0.3
    # eta:
    #   lengthscales/ls_theta: Normal(0.5, 0.25) -> val: mcmc_sample_unconstrained[3] = 0.4
    #   lengthscales/ls_x: Normal(-1,1) -> val: mcmc_sample_unconstrained[4] = 0.5
    #   variances/kernel_var: Normal(1,0.5) -> val: mcmc_sample_unconstrained[5] = 0.6
    # thetas:
    #   calib_param1: Normal(0,1) -> val: mcmc_sample_unconstrained[6] = 0.7
    #   calib_param2: Normal(0,1) -> val: mcmc_sample_unconstrained[7] = 0.8

    # Let's get the actual flattened prior objects to confirm order and distributions
    flat_priors_actual_order = model_params.priors_flat

    expected_lp = (
        flat_priors_actual_order[0].distribution.log_prob(mcmc_sample_unconstrained[0])
        + flat_priors_actual_order[1].distribution.log_prob(
            mcmc_sample_unconstrained[1]
        )
        + flat_priors_actual_order[2].distribution.log_prob(
            mcmc_sample_unconstrained[2]
        )
        + flat_priors_actual_order[3].distribution.log_prob(
            mcmc_sample_unconstrained[3]
        )
        + flat_priors_actual_order[4].distribution.log_prob(
            mcmc_sample_unconstrained[4]
        )
        + flat_priors_actual_order[5].distribution.log_prob(
            mcmc_sample_unconstrained[5]
        )
        + flat_priors_actual_order[6].distribution.log_prob(
            mcmc_sample_unconstrained[6]
        )
        + flat_priors_actual_order[7].distribution.log_prob(
            mcmc_sample_unconstrained[7]
        )
    )

    actual_lp = log_prior_func(mcmc_sample_unconstrained)
    jitted_actual_lp = jitted_log_prior_func(mcmc_sample_unconstrained)

    assert jnp.isclose(actual_lp, expected_lp)
    assert jnp.isclose(jitted_actual_lp, expected_lp)


def test_unflatten_sample(
    model_params: ModelParameters, mcmc_sample_unconstrained: list[float]
):
    unflatten_func = model_params.unflatten_sample
    jitted_unflatten_func = jit(unflatten_func)

    # Based on JAX's alphabetical key sorting for flattening:
    # delta -> epsilon -> eta -> thetas
    # expected_unflattened = {
    #     "delta": {
    #         "lengthscales": {
    #             "ls_x": mcmc_sample_unconstrained[0]
    #         },  # sorted: lengthscales before variances
    #         "variances": {"kernel_var": mcmc_sample_unconstrained[1]},
    #     },
    #     "epsilon": {
    #         "variances": {"obs_noise": mcmc_sample_unconstrained[2]},
    #     },
    #     "eta": {
    #         "lengthscales": {  # sorted: ls_theta before ls_x if keys are "ls_theta", "ls_x"
    #             # if keys are "ls_x", "ls_theta", then order is x then theta.
    #             # valid_prior_dict has ls_x then ls_theta
    #             "ls_theta": mcmc_sample_unconstrained[
    #                 4
    #             ],  # ls_theta is dict key, ls_x is dict key.
    #             "ls_x": mcmc_sample_unconstrained[
    #                 3
    #             ],  # JAX sorts these keys: ls_theta, ls_x. So sample[3] is for ls_theta, sample[4] for ls_x
    #         },
    #         "variances": {"kernel_var": mcmc_sample_unconstrained[5]},
    #     },
    #     "thetas": {  # sorted: calib_param1 before calib_param2
    #         "calib_param1": mcmc_sample_unconstrained[6],
    #         "calib_param2": mcmc_sample_unconstrained[7],
    #     },
    # }
    # Re-evaluating order for eta.lengthscales based on valid_prior_dict:
    # "ls_x": ParameterPrior(npd.Normal(-1,1)),      # param_idx: 3 in its group
    # "ls_theta": ParameterPrior(npd.Normal(0.5,0.25)),# param_idx: 4 in its group
    # So, for eta.lengthscales, 'ls_theta' comes after 'ls_x' if sorted.
    # The mcmc_sample mapping needs to be precise.
    # Let's use model_params.priors_flat to determine the exact mapping for clarity.

    # The actual unflattened tree:
    reconstructed_tree = jtu.tree_unflatten(
        model_params.priors_tree, mcmc_sample_unconstrained
    )

    actual_unflattened = unflatten_func(mcmc_sample_unconstrained)
    jitted_actual_unflattened = jitted_unflatten_func(mcmc_sample_unconstrained)

    assert_trees_equal(
        actual_unflattened, reconstructed_tree
    )  # Compare against programmatically unflattened
    assert_trees_equal(jitted_actual_unflattened, reconstructed_tree)


def test_constrain_sample(
    model_params: ModelParameters, mcmc_sample_unconstrained: list[float]
):
    constrain_func = model_params.constrain_sample
    jitted_constrain_func = jit(constrain_func)
    expected_constrained_flat = mcmc_sample_unconstrained

    actual_constrained_flat = constrain_func(mcmc_sample_unconstrained)
    jitted_actual_constrained_flat = jitted_constrain_func(mcmc_sample_unconstrained)

    assert_trees_equal(actual_constrained_flat, expected_constrained_flat)
    assert_trees_equal(jitted_actual_constrained_flat, expected_constrained_flat)


def test_constrain_and_unflatten_sample(
    model_params: ModelParameters, mcmc_sample_unconstrained: list[float]
):
    constrain_unflatten_func = model_params.constrain_and_unflatten_sample
    jitted_constrain_unflatten_func = jit(constrain_unflatten_func)

    expected_constrained_unflattened = jtu.tree_unflatten(
        model_params.priors_tree, mcmc_sample_unconstrained
    )

    actual_constrained_unflattened = constrain_unflatten_func(mcmc_sample_unconstrained)
    jitted_actual_constrained_unflattened = jitted_constrain_unflatten_func(
        mcmc_sample_unconstrained
    )

    assert_trees_equal(actual_constrained_unflattened, expected_constrained_unflattened)
    assert_trees_equal(
        jitted_actual_constrained_unflattened, expected_constrained_unflattened
    )


def test_model_parameters_empty_prior_dict_fails():
    """Tests ModelParameters with an empty prior_dict raises ValueError due to _check_prior_dict."""
    empty_prior_dict: ModelParameterDict = {}
    with pytest.raises(
        ValueError, match="prior_dict keys must contain \\['thetas', 'eta', 'delta'\\]"
    ):
        ModelParameters(empty_prior_dict)


def test_model_parameters_alternative_priors_valid_structure():
    """Tests ModelParameters with different distributions but valid top-level structure."""
    alt_prior_dict: ModelParameterDict = {
        "thetas": {"calib": ParameterPrior(npd.Normal(0, 1))},
        "eta": {
            "variances": {
                "k_var": ParameterPrior(npd.Exponential(1.0))
            },  # Non-Identity
            "lengthscales": {"ls": ParameterPrior(npd.HalfNormal(1.0))},  # Non-Identity
        },
        "delta": {
            "variances": {"d_var": ParameterPrior(npd.LogNormal(0, 1))},  # Non-Identity
            "lengthscales": {"d_ls": ParameterPrior(npd.Normal(0, 2))},  # Identity
        },
    }
    params = ModelParameters(alt_prior_dict)
    assert (
        params.n_params == 5
    )  # 1 (thetas) + 1 (eta/var) + 1 (eta/ls) + 1 (delta/var) + 1 (delta/ls)

    bijector_names = sorted([p.bijector.__class__.__name__ for p in params.priors_flat])
    expected_bijector_names = sorted(
        [
            "ExpTransform",  # delta/variances (LogNormal)
            "IdentityTransform",  # delta/lengthscales (Normal)
            "ExpTransform",  # eta/variances (Exponential)
            "ExpTransform",  # eta/lengthscales (HalfNormal)
            "IdentityTransform",  # thetas/calib (Normal)
        ]
    )
    assert bijector_names == expected_bijector_names

    unconstrained_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    log_prior_func = params.get_log_prior_func()
    assert isinstance(log_prior_func(unconstrained_vals), jnp.ndarray)  # Check it runs

    constrained_flat = params.constrain_sample(unconstrained_vals)
    assert len(constrained_flat) == 5

    # Check positivity for non-identity transforms
    # Order of params.priors_flat: (delta (ls,var), eta (ls,var), thetas (calib))
    # delta.lengthscales.d_ls (Normal) -> Id
    # delta.variances.d_var (LogNormal) -> Exp
    # eta.lengthscales.ls (HalfNormal) -> Exp
    # eta.variances.k_var (Exponential) -> Exp
    # thetas.calib (Normal) -> Id

    # Get actual order:
    # idx_map = {name: i for i, name in enumerate(jtu.tree_leaves(jtu.tree_map_with_path(lambda path, x: path[-1].key, params.priors)))}
    # This is getting complicated to map specific unconstrained_vals to specific priors without running it.
    # Instead, check general properties:
    for i, prior_item in enumerate(params.priors_flat):
        constrained_val = constrained_flat[i]
        unconstrained_val = unconstrained_vals[i]
        if isinstance(
            prior_item.distribution, (npd.Exponential, npd.HalfNormal, npd.LogNormal)
        ):
            assert constrained_val > 0, (
                f"Prior {prior_item} gave non-positive value {constrained_val}"
            )
        elif isinstance(prior_item.distribution, npd.Normal):
            assert jnp.isclose(constrained_val, unconstrained_val), (
                f"Prior {prior_item} (Normal) did not return identity: {constrained_val} vs {unconstrained_val}"
            )


def test_model_parameters_pytree_registration(model_params: ModelParameters):
    assert len(model_params.priors_flat) == model_params.n_params
    reconstructed_priors = jtu.tree_unflatten(
        model_params.priors_tree, model_params.priors_flat
    )

    original_flat_priors_info = [
        (type(p.distribution), p.distribution.arg_constraints)
        for p in model_params.priors_flat
    ]
    reconstructed_flat, _ = jtu.tree_flatten(
        reconstructed_priors
    )  # reconstructed_priors is already the dict
    reconstructed_flat_priors = jtu.tree_leaves(
        reconstructed_priors, is_leaf=lambda x: isinstance(x, ParameterPrior)
    )

    reconstructed_flat_priors_info = [
        (type(p.distribution), p.distribution.arg_constraints)
        for p in reconstructed_flat_priors
    ]

    # Sort them before comparing as flatten order might differ if dicts were created differently
    assert sorted(original_flat_priors_info, key=str) == sorted(
        reconstructed_flat_priors_info, key=str
    )


def test_parameter_prior_non_identity_bijector():
    prior = ParameterPrior(npd.Exponential(rate=1.0))
    assert prior.bijector.__class__.__name__ == "ExpTransform"

    unconstrained_val = 0.5
    constrained_val = prior.forward(unconstrained_val)
    expected_constrained_val = jnp.exp(0.5)
    assert jnp.isclose(constrained_val, expected_constrained_val)

    expected_log_prob = (
        npd.Exponential(rate=1.0).log_prob(jnp.exp(unconstrained_val))
        + unconstrained_val
    )
    actual_log_prob = prior.log_prob(unconstrained_val)
    assert jnp.isclose(actual_log_prob, expected_log_prob)
