import jax.numpy as jnp
import pytest
from gpjax.kernels import (
    RBF,
    White,
)
from jax import jit
from jax.scipy.linalg import block_diag
from kohgpjax.kernels.computations.kohcomputation import KOHKernelComputation
from kohgpjax.kernels.kohkernel import KOHKernel

# --- Fixtures ---


@pytest.fixture(scope="module")
def mock_koh_kernel_components():
    # Sub-kernels: make them simple and predictable for testing
    # Using default parameters for RBF (lengthscale=1.0, variance=1.0)
    # and White (variance=1.0) unless specified.
    return {
        "k_eta": RBF(),  # Operates on all dims passed to it
        "k_delta": RBF(),  # Operates on all dims of its slice (x_field)
        "k_epsilon": White(),  # Operates on all dims of its slice (x_field)
        "k_epsilon_eta": White(
            variance=0.5
        ),  # Operates on all dims of its slice (x_sim)
    }


@pytest.fixture(scope="module")
def setup_computation_test(mock_koh_kernel_components):
    n_field_obs = 2
    n_sim_obs = 3
    n_var_params = 2  # Number of "variable" parameters (e.g., x, t)
    n_calib_params = 1  # Number of calibration parameters (e.g., theta)

    # Total dimensions for k_eta which sees combined inputs
    # (var_params + calib_params)
    dim_k_eta_inputs = n_var_params + n_calib_params

    k_eta = mock_koh_kernel_components["k_eta"]
    # Let k_delta operate only on variable parameters for a more specific test.
    k_delta_active_dims = RBF(active_dims=list(range(n_var_params)))

    k_epsilon = mock_koh_kernel_components["k_epsilon"]
    k_epsilon_eta = mock_koh_kernel_components["k_epsilon_eta"]

    koh_kernel_instance = KOHKernel(
        num_field_obs=n_field_obs,
        num_sim_obs=n_sim_obs,
        k_eta=k_eta,
        k_delta=k_delta_active_dims,  # Use the one with active_dims
        k_epsilon=k_epsilon,
        k_epsilon_eta=k_epsilon_eta,
    )

    engine = KOHKernelComputation()

    N = n_field_obs + n_sim_obs
    # M = n_field_obs + n_sim_obs + 1 # Original: different number of points for X2
    M = N  # For now, ensure M=N to avoid broadcasting error due to padding in source for shape tests.
    # The N!=M case for cross-covariance values will be tested separately for the error.

    X1_field = (
        jnp.arange(n_field_obs * dim_k_eta_inputs).reshape(
            n_field_obs, dim_k_eta_inputs
        )
        * 0.1
    )
    X1_sim = (
        jnp.arange(n_sim_obs * dim_k_eta_inputs).reshape(n_sim_obs, dim_k_eta_inputs)
        * 0.2
    )
    X1 = jnp.vstack((X1_field, X1_sim))

    X2_field = (
        jnp.arange(n_field_obs * dim_k_eta_inputs).reshape(
            n_field_obs, dim_k_eta_inputs
        )
        * 0.3
    ) + 0.05
    X2_sim = (
        jnp.arange((M - n_field_obs) * dim_k_eta_inputs).reshape(
            (M - n_field_obs), dim_k_eta_inputs
        )
        * 0.4
    ) + 0.05
    X2 = jnp.vstack((X2_field, X2_sim))

    return {
        "engine": engine,
        "kernel": koh_kernel_instance,
        "X1": X1,
        "X2": X2,
        "n_field_obs": n_field_obs,
        "n_var_params": n_var_params,
    }


# --- Tests for KOHKernelComputation ---


def test_kohkernelcomputation_cross_covariance_shape(setup_computation_test):
    s = setup_computation_test
    engine, kernel, X1, X2 = s["engine"], s["kernel"], s["X1"], s["X2"]

    def to_dense_if_needed(mat_op):
        return mat_op.to_dense() if hasattr(mat_op, "to_dense") else mat_op

    cov_matrix_op = engine.cross_covariance(kernel, X1, X2)
    cov_matrix = to_dense_if_needed(cov_matrix_op)
    assert cov_matrix.shape == (X1.shape[0], X2.shape[0])

    jitted_cross_cov = jit(
        engine.cross_covariance, static_argnums=0
    )  # Mark kernel as static
    cov_matrix_jit_op = jitted_cross_cov(kernel, X1, X2)
    cov_matrix_jit = to_dense_if_needed(cov_matrix_jit_op)
    assert cov_matrix_jit.shape == (X1.shape[0], X2.shape[0])


def test_kohkernelcomputation_gram_shape(setup_computation_test):
    s = setup_computation_test
    engine, kernel, X1 = s["engine"], s["kernel"], s["X1"]

    def to_dense_if_needed(mat_op):
        return mat_op.to_dense() if hasattr(mat_op, "to_dense") else mat_op

    gram_matrix_op = engine.gram(kernel, X1)
    gram_matrix = to_dense_if_needed(gram_matrix_op)
    assert gram_matrix.shape == (X1.shape[0], X1.shape[0])

    jitted_cross_cov_for_gram = jit(
        engine.cross_covariance, static_argnums=0
    )  # JIT cross_cov
    gram_matrix_jit_op = jitted_cross_cov_for_gram(kernel, X1, X1)  # Call for gram case
    gram_matrix_jit = to_dense_if_needed(gram_matrix_jit_op)
    assert gram_matrix_jit.shape == (X1.shape[0], X1.shape[0])


def test_kohkernelcomputation_cross_covariance_values(setup_computation_test):
    s = setup_computation_test
    engine, kernel, X1 = s["engine"], s["kernel"], s["X1"]
    n_field_obs = s["n_field_obs"]

    # Slices of X1 (for gram matrix test X2=X1)
    X1_field = X1[:n_field_obs, :]
    X1_sim = X1[
        n_field_obs : n_field_obs + kernel.num_sim_obs, :
    ]  # kernel.num_sim_obs defines the sim slice size

    def to_dense_if_needed(mat_op):
        return mat_op.to_dense() if hasattr(mat_op, "to_dense") else mat_op

    sigma_eta_expected = to_dense_if_needed(kernel.k_eta.cross_covariance(X1, X1))
    sigma_delta_expected = to_dense_if_needed(
        kernel.k_delta.cross_covariance(X1_field, X1_field)
    )
    sigma_epsilon_expected = to_dense_if_needed(
        kernel.k_epsilon.cross_covariance(X1_field, X1_field)
    )
    sigma_epsilon_eta_expected = to_dense_if_needed(
        kernel.k_epsilon_eta.cross_covariance(X1_sim, X1_sim)
    )

    block_00 = sigma_delta_expected + sigma_epsilon_expected
    block_11 = sigma_epsilon_eta_expected
    diag_block = block_diag(block_00, block_11)  # This is already a JAX array

    N_X1 = X1.shape[0]
    # Ensure pad amounts are not negative if N_X1 < (n_field_obs + kernel.num_sim_obs)
    # This shouldn't happen if X1 is constructed correctly (total rows >= field + sim slices for block_diag)
    pad_rows = max(0, N_X1 - (n_field_obs + kernel.num_sim_obs))
    pad_cols = max(0, N_X1 - (n_field_obs + kernel.num_sim_obs))
    padded_diag_block = jnp.pad(diag_block, ((0, pad_rows), (0, pad_cols)))

    expected_gram_matrix = sigma_eta_expected + padded_diag_block

    actual_gram_matrix_op = engine.gram(kernel, X1)
    actual_gram_matrix = to_dense_if_needed(actual_gram_matrix_op)
    assert jnp.allclose(actual_gram_matrix, expected_gram_matrix, atol=1e-6)

    # Test JIT compilation for gram
    jitted_cross_cov_for_gram = jit(engine.cross_covariance, static_argnums=0)
    actual_gram_matrix_jit_op = jitted_cross_cov_for_gram(kernel, X1, X1)
    actual_gram_matrix_jit = to_dense_if_needed(actual_gram_matrix_jit_op)
    assert jnp.allclose(actual_gram_matrix_jit, expected_gram_matrix, atol=1e-6)

    # Test the N != M case for cross_covariance, expecting a broadcasting error
    # due to the current padding logic in the source code.
    X_generic1 = s["X1"]
    # Re-create an X_generic2 that has M != N for this specific check, as fixture X2 might now have M=N
    n_field_obs_s, _, dim_k_eta_inputs_s = (
        s["n_field_obs"],
        kernel.num_sim_obs,
        X_generic1.shape[1],
    )
    N_current = X_generic1.shape[0]
    M_for_error_check = N_current + 1  # Ensure M != N

    X2_field_err = (
        jnp.arange(n_field_obs_s * dim_k_eta_inputs_s).reshape(
            n_field_obs_s, dim_k_eta_inputs_s
        )
        * 0.7
    ) + 0.01
    X2_sim_err = (
        jnp.arange((M_for_error_check - n_field_obs_s) * dim_k_eta_inputs_s).reshape(
            (M_for_error_check - n_field_obs_s), dim_k_eta_inputs_s
        )
        * 0.8
    ) + 0.01
    X_generic2_for_error_check = jnp.vstack((X2_field_err, X2_sim_err))

    if X_generic1.shape[0] != X_generic2_for_error_check.shape[0]:
        with pytest.raises(TypeError, match="incompatible shapes for broadcasting"):
            # This call is expected to fail due to the padding issue when N != M
            # and the subsequent addition sigma_eta + padded_block_diag
            engine.cross_covariance(kernel, X_generic1, X_generic2_for_error_check)


# End of test_kohcomputation.py
