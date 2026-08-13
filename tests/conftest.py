import jax.random as jr
import pytest
from jax import config


def pytest_configure():
    """Enable the suite's documented float64 numerical contract at startup."""
    config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def key():
    """Provides a session-scoped JAX PRNG key for tests requiring randomness."""
    return jr.key(42)


# End of conftest.py
