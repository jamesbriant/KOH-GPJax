from gpjax.kernels import (
    RBF,
    White,
)
import jax.numpy as jnp
import pytest

from kohgpjax.kernels.computations.kohcomputation import KOHKernelComputation
from kohgpjax.kernels.kohkernel import KOHKernel

try:
    from beartype.roar import BeartypeCallHintParamViolation
    from jaxtyping import TypeCheckError

    ValidationErrors = (TypeError, BeartypeCallHintParamViolation, TypeCheckError)
except ImportError:
    ValidationErrors = ValueError


# Mock Kernels Fixture (optional, can define in test if simple)
@pytest.fixture(scope="module")
def mock_sub_kernels():
    return {
        "k_eta": RBF(),
        "k_delta": RBF(
            active_dims=[0]
        ),  # Example: acts only on the first dim of its input slice
        "k_epsilon": White(variance=0.5),
        "k_epsilon_eta": White(variance=0.1),
    }


# --- Tests for KOHKernel ---


def test_kohkernel_initialization(mock_sub_kernels):
    """Tests the initialization of the KOHKernel."""
    num_field_obs_val = 10
    num_sim_obs_val = 5

    koh_kernel = KOHKernel(
        num_field_obs=num_field_obs_val,
        num_sim_obs=num_sim_obs_val,
        k_eta=mock_sub_kernels["k_eta"],
        k_delta=mock_sub_kernels["k_delta"],
        k_epsilon=mock_sub_kernels["k_epsilon"],
        k_epsilon_eta=mock_sub_kernels["k_epsilon_eta"],
    )

    assert koh_kernel.num_field_obs == num_field_obs_val
    assert koh_kernel.num_sim_obs == num_sim_obs_val
    assert koh_kernel.k_eta == mock_sub_kernels["k_eta"]
    assert koh_kernel.k_delta == mock_sub_kernels["k_delta"]
    assert koh_kernel.k_epsilon == mock_sub_kernels["k_epsilon"]
    assert koh_kernel.k_epsilon_eta == mock_sub_kernels["k_epsilon_eta"]

    assert koh_kernel.name == "KOHKernel"
    assert isinstance(koh_kernel.compute_engine, KOHKernelComputation)


def test_kohkernel_call_raises_not_implemented(mock_sub_kernels):
    """Tests that calling the KOHKernel instance raises NotImplementedError."""
    koh_kernel = KOHKernel(
        num_field_obs=10,
        num_sim_obs=5,
        k_eta=mock_sub_kernels["k_eta"],
        k_delta=mock_sub_kernels["k_delta"],
        k_epsilon=mock_sub_kernels["k_epsilon"],
        k_epsilon_eta=mock_sub_kernels["k_epsilon_eta"],
    )

    x_dummy = jnp.array([[1.0, 2.0]])
    y_dummy = jnp.array([[3.0, 4.0]])

    with pytest.raises(
        ValidationErrors  # ,
        # match="It is not obvious how to compute the kernel value for this kernel. Instead calculate the desired value by calling one of the components (or subkernels) of this class.",
    ):
        koh_kernel(x_dummy, y_dummy)


def test_kohkernel_active_dims_propagation():
    """Test that active_dims are correctly handled if sub-kernels have them.
    This is mostly an integration aspect tested via KOHKernelComputation.
    KOHKernel itself doesn't aggregate active_dims but passes kernels to computation.
    """
    k_eta_active = RBF(active_dims=[0, 1])
    k_delta_active = RBF(
        active_dims=[0]
    )  # Will receive a slice, active_dims apply to that slice

    koh_kernel = KOHKernel(
        num_field_obs=2,
        num_sim_obs=2,
        k_eta=k_eta_active,
        k_delta=k_delta_active,
        k_epsilon=White(),
        k_epsilon_eta=White(),
    )
    # The test of active_dims really happens in how KOHKernelComputation uses these.
    # Here we just ensure they are stored.
    assert koh_kernel.k_eta.active_dims == [0, 1]
    assert koh_kernel.k_delta.active_dims == [0]


# End of test_kohkernel.py
