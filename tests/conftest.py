import pytest
from jax import config
import jax.random as jr

@pytest.fixture(autouse=True)
def enable_x64_for_all_tests():
    """Ensures JAX float64 precision is enabled for all tests."""
    config.update("jax_enable_x64", True)
    # No yield needed as this is a configuration applied globally before tests run.

@pytest.fixture(scope="session")
def key():
    """Provides a session-scoped JAX PRNG key for tests requiring randomness."""
    return jr.key(42)

# End of conftest.py
