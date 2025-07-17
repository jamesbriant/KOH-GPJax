import gpjax as gpx
import pytest
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.parameters import Static
from jax import (  # Removed 'config'
    jit,
)
from jax import (
    numpy as jnp,
)
from jax import (
    tree_util as jtu,
)
from kohgpjax.gps import (
    KOHPosterior,
    construct_posterior,
)
from kohgpjax.kernels.kohkernel import KOHKernel  # Needed for the prior fixture

# --- Fixtures ---


@pytest.fixture(scope="module")
def mock_koh_kernel_for_prior() -> KOHKernel:
    # Minimal KOHKernel for prior construction
    return KOHKernel(
        num_field_obs=2,
        num_sim_obs=2,
        k_eta=gpx.kernels.RBF(),
        k_delta=gpx.kernels.RBF(),
        k_epsilon=gpx.kernels.White(),
        k_epsilon_eta=gpx.kernels.White(
            variance=0.01
        ),  # Small non-zero for distinctness
    )


@pytest.fixture(scope="module")
def mock_prior_fixture(mock_koh_kernel_for_prior: KOHKernel) -> gpx.gps.Prior:
    return gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=mock_koh_kernel_for_prior,  # Crucial for getting KOHPosterior from construct_posterior
    )


@pytest.fixture(scope="module")
def mock_standard_prior_fixture() -> gpx.gps.Prior:  # For testing non-KOH path
    return gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
    )


@pytest.fixture(scope="module")
def mock_likelihood_fixture() -> gpx.likelihoods.Gaussian:
    # num_datapoints should match dataset used for predictions
    return gpx.likelihoods.Gaussian(
        num_datapoints=Static(jnp.array(4))
    )  # Example: 2 field + 2 sim


@pytest.fixture(scope="module")
def mock_dataset_fixture() -> Dataset:
    # Dataset for prediction methods: 2 field + 2 sim = 4 total
    # Dimensions should be compatible with kernels if doing value checks,
    # but for type/run checks, simple data is fine.
    # KOHKernel's sub-kernels might have active_dims. Assume default RBFs operate on all dims.
    # X needs to match num_datapoints in likelihood if used for posterior construction,
    # but here it's for prediction.
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])  # (4, 2)
    y = jnp.array([[1.0], [2.0], [3.0], [4.0]])  # (4, 1)
    return Dataset(X=X, y=y)


@pytest.fixture(scope="module")
def test_input_points_fixture() -> jnp.ndarray:
    return jnp.array([[0.5, 1.5], [2.5, 3.5]])  # (2, 2)


# --- Tests for construct_posterior ---


def test_construct_posterior_returns_kohposterior(
    mock_prior_fixture: gpx.gps.Prior,  # This prior uses KOHKernel
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)
    assert isinstance(posterior, KOHPosterior)
    assert posterior.prior == mock_prior_fixture
    assert posterior.likelihood == mock_likelihood_fixture
    # 'name' attribute is not standard on AbstractPosterior or its subclasses like ConjugatePosterior/KOHPosterior in GPJax


def test_construct_posterior_returns_conjugateposterior(
    mock_standard_prior_fixture: gpx.gps.Prior,  # This prior uses RBF kernel
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
):
    posterior = construct_posterior(
        mock_standard_prior_fixture, mock_likelihood_fixture
    )
    assert isinstance(posterior, gpx.gps.ConjugatePosterior)
    assert not isinstance(posterior, KOHPosterior)  # Should not be KOHPosterior
    assert posterior.prior == mock_standard_prior_fixture
    assert posterior.likelihood == mock_likelihood_fixture


# --- Tests for KOHPosterior ---


def test_koh_posterior_inheritance():
    assert issubclass(KOHPosterior, gpx.gps.AbstractPosterior)
    # While it behaves like a ConjugatePosterior, its direct inheritance is AbstractPosterior
    # The functionality for conjugate updates is what matters.


def test_koh_posterior_predict_raises_not_implemented(
    mock_prior_fixture: gpx.gps.Prior,
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
    test_input_points_fixture: jnp.ndarray,
    mock_dataset_fixture: Dataset,
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)
    with pytest.raises(
        NotImplementedError,
        match="Use predict_eta\\(\\), predict_zeta\\(\\) or predict_obs\\(\\) methods instead\\.",
    ):
        posterior.predict(test_input_points_fixture, train_data=mock_dataset_fixture)


def test_koh_posterior_predict_eta_type(
    mock_prior_fixture: gpx.gps.Prior,
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
    test_input_points_fixture: jnp.ndarray,
    mock_dataset_fixture: Dataset,
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)
    prediction = posterior.predict_eta(
        test_input_points_fixture, train_data=mock_dataset_fixture
    )
    assert isinstance(prediction, GaussianDistribution)
    # Check shapes based on test_input_points_fixture
    assert prediction.mean.shape == (
        test_input_points_fixture.shape[0],
    )  # Mean is 1D if output dim is 1
    assert prediction.covariance().shape == (
        test_input_points_fixture.shape[0],
        test_input_points_fixture.shape[0],
    )


def test_koh_posterior_predict_zeta_type(
    mock_prior_fixture: gpx.gps.Prior,
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
    test_input_points_fixture: jnp.ndarray,
    mock_dataset_fixture: Dataset,
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)
    # Test with include_observation_noise = False (default)
    prediction_false = posterior.predict_zeta(
        test_input_points_fixture,
        train_data=mock_dataset_fixture,
        include_observation_noise=False,
    )
    assert isinstance(prediction_false, GaussianDistribution)
    assert prediction_false.mean.shape == (
        test_input_points_fixture.shape[0],
    )  # Mean is 1D
    assert prediction_false.covariance().shape == (
        test_input_points_fixture.shape[0],
        test_input_points_fixture.shape[0],
    )

    # Test with include_observation_noise = True
    prediction_true = posterior.predict_zeta(
        test_input_points_fixture,
        train_data=mock_dataset_fixture,
        include_observation_noise=True,
    )
    assert isinstance(prediction_true, GaussianDistribution)
    assert prediction_true.mean.shape == (
        test_input_points_fixture.shape[0],
    )  # Mean is 1D
    assert prediction_true.covariance().shape == (
        test_input_points_fixture.shape[0],
        test_input_points_fixture.shape[0],
    )

    # Check JIT compilability
    # JIT with static include_observation_noise works for both False and True
    jitted_predict_zeta_static = jit(
        posterior.predict_zeta, static_argnames="include_observation_noise"
    )

    prediction_jit_false = jitted_predict_zeta_static(
        test_input_points_fixture,
        train_data=mock_dataset_fixture,
        include_observation_noise=False,
    )
    assert isinstance(prediction_jit_false, GaussianDistribution)

    prediction_jit_true = jitted_predict_zeta_static(
        test_input_points_fixture,
        train_data=mock_dataset_fixture,
        include_observation_noise=True,
    )
    assert isinstance(prediction_jit_true, GaussianDistribution)


def test_koh_posterior_predict_obs_type(
    mock_prior_fixture: gpx.gps.Prior,
    mock_likelihood_fixture: gpx.likelihoods.Gaussian,
    test_input_points_fixture: jnp.ndarray,
    mock_dataset_fixture: Dataset,
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)
    prediction = posterior.predict_obs(
        test_input_points_fixture, train_data=mock_dataset_fixture
    )
    assert isinstance(prediction, GaussianDistribution)
    assert prediction.mean.shape == (test_input_points_fixture.shape[0],)  # Mean is 1D
    assert prediction.covariance().shape == (
        test_input_points_fixture.shape[0],
        test_input_points_fixture.shape[0],
    )


def test_koh_posterior_pytree(
    mock_prior_fixture: gpx.gps.Prior, mock_likelihood_fixture: gpx.likelihoods.Gaussian
):
    posterior = construct_posterior(mock_prior_fixture, mock_likelihood_fixture)

    leaves, treedef = jtu.tree_flatten(posterior)
    reconstructed_posterior = jtu.tree_unflatten(treedef, leaves)

    assert isinstance(reconstructed_posterior, KOHPosterior)
    assert reconstructed_posterior.prior == posterior.prior
    assert reconstructed_posterior.likelihood == posterior.likelihood
    # Compare some properties to ensure it's a valid reconstruction
    # Removed .name assertion as it's not a standard attribute
    assert (
        reconstructed_posterior.prior.kernel.num_field_obs
        == posterior.prior.kernel.num_field_obs
    )


# End of test_gps.py
