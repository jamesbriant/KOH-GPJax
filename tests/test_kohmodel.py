import gpjax as gpx  # Using gpx alias for gpjax
import numpyro.distributions as npd
import pytest
from flax import nnx
from gpjax.dataset import Dataset  # Corrected import
from gpjax.parameters import Parameter
from jax import config
from jax import (  # Removed 'config'
    jit,
)
from jax import (
    numpy as jnp,
)
from kohgpjax.dataset import KOHDataset
from kohgpjax.gps import KOHPosterior  # Assuming this is the correct import path
from kohgpjax.kernels.kohkernel import KOHKernel
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import (
    ModelParameterDict,
    ModelParameterPriorDict,
    ModelParameters,
    ParameterPrior,
)

config.update("jax_enable_x64", True)

# --- Minimal Concrete KOHModel Subclass ---


class MinimalKOHModel(KOHModel):
    def k_eta(self, params_constrained: dict) -> gpx.kernels.AbstractKernel:
        # Uses 'eta_ls' and 'eta_var' from params_constrained if available, else defaults
        ls = params_constrained.get("eta", {}).get("lengthscales", {}).get("ls", 1.0)
        var = params_constrained.get("eta", {}).get("variances", {}).get("var", 1.0)
        return gpx.kernels.RBF(lengthscale=ls, variance=var)

    def k_delta(self, params_constrained: dict) -> gpx.kernels.AbstractKernel:
        ls = params_constrained.get("delta", {}).get("lengthscales", {}).get("ls", 1.0)
        var = params_constrained.get("delta", {}).get("variances", {}).get("var", 1.0)
        return gpx.kernels.RBF(
            lengthscale=ls,
            variance=var,
            active_dims=list(
                range(self._kohdataset_cached_for_test.num_variable_params)
            ),
        )

    # Cache kohdataset for k_delta's active_dims setting during tests
    # This is a workaround for tests as k_delta in real scenarios might get num_variable_params differently
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kohdataset_cached_for_test = kwargs.get("kohdataset")


class MissingEtaKOHModel(KOHModel):
    def k_delta(self, params_constrained: dict) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.RBF()


class MissingDeltaKOHModel(KOHModel):
    def k_eta(self, params_constrained: dict) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.RBF()


# --- Fixtures ---


@pytest.fixture(scope="module")
def simple_prior_dict(
    koh_dataset_fixture,
) -> ModelParameterPriorDict:  # Depends on koh_dataset_fixture for num_calib_params
    num_calib = koh_dataset_fixture.num_calib_params
    thetas_priors = {
        f"theta_{i}": ParameterPrior(npd.Normal(0, 1)) for i in range(num_calib)
    }

    return {
        "thetas": thetas_priors,
        "eta": {
            "variances": {
                "var": ParameterPrior(npd.LogNormal(0, 1))
            },  # For RBF variance
            "lengthscales": {
                "ls": ParameterPrior(npd.LogNormal(0, 1))
            },  # For RBF lengthscale
        },
        "delta": {
            "variances": {"var": ParameterPrior(npd.LogNormal(0, 1))},
            "lengthscales": {"ls": ParameterPrior(npd.LogNormal(0, 1))},
        },
        "epsilon": {
            "variances": {"obs_noise": ParameterPrior(npd.HalfNormal(1.0))}
        },  # For k_epsilon if obs_stddev is None
    }


@pytest.fixture(scope="module")
def model_parameters_fixture(
    simple_prior_dict: ModelParameterPriorDict,
) -> ModelParameters:
    return ModelParameters(simple_prior_dict)


@pytest.fixture(scope="module")
def koh_dataset_fixture() -> KOHDataset:
    # num_variable_params = 1, num_calib_params = 1
    x_field = jnp.array([[1.0], [2.0]])  # 2 obs, 1 var_param
    y_field = jnp.array([[10.0], [20.0]])
    field_ds = Dataset(x_field, y_field)

    # x_sim: first col = var_param, second col = design_param for sim
    # num_sim_inputs = var_params + design_params = 1 + 1 = 2
    # num_calib_params = 1, so total for KOHDataset X_sim is var_params + calib_params
    # The structure for KOHDataset's sim_dataset.X should be [var_params, design_params]
    # And KOHDataset derives num_calib_params from sim_dataset.X.shape[1] - field_dataset.X.shape[1]
    # So, if field_dataset.X.shape[1] is 1 (var_params)
    # and sim_dataset.X.shape[1] is 2 (var_params, calib_params_design_for_sim)
    # then num_calib_params will be 1.
    x_sim = jnp.array(
        [[1.0, 0.1], [2.0, 0.2]]
    )  # 2 obs, 1 var_param, 1 "design" param for sim (becomes calib)
    y_sim = jnp.array([[11.0], [22.0]])
    sim_ds = Dataset(x_sim, y_sim)

    return KOHDataset(field_ds, sim_ds)


@pytest.fixture(scope="module")
def gpjax_params_fixture(
    model_parameters_fixture: ModelParameters,
) -> ModelParameterDict:
    # Create a dummy flat sample of unconstrained params
    # Number of params from simple_prior_dict:
    # thetas (1) + eta.var (1) + eta.ls (1) + delta.var (1) + delta.ls (1) + epsilon.var (1) = 6
    num_params = model_parameters_fixture.n_params
    dummy_flat_unconstrained = jnp.zeros(num_params)
    return model_parameters_fixture.constrain_and_unflatten_sample(
        dummy_flat_unconstrained
    )


@pytest.fixture
def minimal_koh_model_fixture(
    model_parameters_fixture: ModelParameters, koh_dataset_fixture: KOHDataset
) -> MinimalKOHModel:
    return MinimalKOHModel(
        model_parameters=model_parameters_fixture,
        kohdataset=koh_dataset_fixture,
        jitter=1e-6,
    )


@pytest.fixture
def minimal_koh_model_static_obs_fixture(
    model_parameters_fixture: ModelParameters, koh_dataset_fixture: KOHDataset
) -> MinimalKOHModel:
    return MinimalKOHModel(
        model_parameters=model_parameters_fixture,
        kohdataset=koh_dataset_fixture,
        obs_stddev=jnp.array([0.1]),  # fixed obs_stddev
        jitter=1e-6,
    )


# --- Tests ---


def test_kohmodel_initialization(
    minimal_koh_model_fixture: MinimalKOHModel,
    model_parameters_fixture,
    koh_dataset_fixture,
):
    model = minimal_koh_model_fixture
    assert model.model_parameters == model_parameters_fixture
    assert model.kohdataset == koh_dataset_fixture
    assert model.jitter == 1e-6
    assert model.obs_var is None  # obs_stddev was None

    model_static = MinimalKOHModel(
        model_parameters_fixture,
        koh_dataset_fixture,
        obs_stddev=jnp.array([0.5]),
    )
    assert jnp.isclose(model_static.obs_var, 0.5**2)


def test_gp_prior_mean_function(minimal_koh_model_fixture: MinimalKOHModel):
    assert isinstance(
        minimal_koh_model_fixture.GP_prior_mean_function(), gpx.mean_functions.Zero
    )


def test_k_epsilon(
    minimal_koh_model_fixture: MinimalKOHModel,
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: dict,
    koh_dataset_fixture: KOHDataset,
):
    # Case 1: obs_stddev was None, variance from params
    k_eps_dynamic = minimal_koh_model_fixture.k_epsilon(gpjax_params_fixture)
    assert isinstance(k_eps_dynamic, gpx.kernels.White)
    # Variance should be taken from gpjax_params_fixture['epsilon']['variances']['obs_noise']
    # which is npd.HalfNormal(1.0).log_prob(0.0) -> constrained value is exp(0) = 1.0
    expected_dyn_var = gpjax_params_fixture["epsilon"]["variances"]["obs_noise"]
    assert jnp.isclose(k_eps_dynamic.variance.value, expected_dyn_var)
    assert isinstance(k_eps_dynamic.variance, gpx.parameters.Parameter)
    assert k_eps_dynamic.active_dims == list(
        range(koh_dataset_fixture.num_variable_params)
    )

    # Case 2: obs_stddev was static
    k_eps_static = minimal_koh_model_static_obs_fixture.k_epsilon(gpjax_params_fixture)
    assert isinstance(k_eps_static, gpx.kernels.White)
    assert not isinstance(k_eps_static.variance, gpx.parameters.Parameter)
    assert jnp.isclose(k_eps_static.variance.value, 0.1**2)
    assert k_eps_static.active_dims == list(
        range(koh_dataset_fixture.num_variable_params)
    )


@pytest.mark.parametrize(
    "invalid_obs_stddev", [-0.1, jnp.array([-0.1]), jnp.array([0.1, 0.2]), object()]
)
def test_kohmodel_rejects_invalid_obs_stddev_values(
    model_parameters_fixture: ModelParameters,
    koh_dataset_fixture: KOHDataset,
    invalid_obs_stddev,
):
    with pytest.raises(
        ValueError,
        match=r"`obs_stddev` must be",
    ):
        MinimalKOHModel(
            model_parameters=model_parameters_fixture,
            kohdataset=koh_dataset_fixture,
            obs_stddev=invalid_obs_stddev,
        )


def test_k_epsilon_requires_dynamic_variance(
    minimal_koh_model_fixture: MinimalKOHModel,
):
    with pytest.raises(
        ValueError,
        match=r"k_epsilon: variance is None from both self\.obs_var and params_constrained\.",
    ):
        minimal_koh_model_fixture.k_epsilon({})


def test_static_and_learned_observation_noise_have_distinct_numeric_covariances(
    minimal_koh_model_fixture: MinimalKOHModel,
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
    koh_dataset_fixture: KOHDataset,
):
    learned_params = {
        **gpjax_params_fixture,
        "epsilon": {
            **gpjax_params_fixture["epsilon"],
            "variances": {"obs_noise": jnp.array(0.25)},
        },
    }
    learned_kernel = minimal_koh_model_fixture.k_epsilon(learned_params)
    static_kernel = minimal_koh_model_static_obs_fixture.k_epsilon(
        gpjax_params_fixture
    )

    learned_covariance = learned_kernel.cross_covariance(
        koh_dataset_fixture.Xf, koh_dataset_fixture.Xf
    )
    static_covariance = static_kernel.cross_covariance(
        koh_dataset_fixture.Xf, koh_dataset_fixture.Xf
    )

    assert jnp.allclose(learned_covariance, jnp.eye(koh_dataset_fixture.num_field_obs) * 0.25)
    assert jnp.allclose(static_covariance, jnp.eye(koh_dataset_fixture.num_field_obs) * 0.01)
    assert not jnp.allclose(learned_covariance, static_covariance)


def test_fixed_observation_noise_is_independent_of_model_parameters(
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
    koh_dataset_fixture: KOHDataset,
):
    changed_params = {
        **gpjax_params_fixture,
        "epsilon": {
            **gpjax_params_fixture["epsilon"],
            "variances": {"obs_noise": jnp.array(100.0)},
        },
    }
    X = koh_dataset_fixture.Xf

    fixed_covariance = minimal_koh_model_static_obs_fixture.k_epsilon(
        gpjax_params_fixture
    ).cross_covariance(X, X)
    changed_covariance = minimal_koh_model_static_obs_fixture.k_epsilon(
        changed_params
    ).cross_covariance(X, X)

    assert jnp.allclose(fixed_covariance, changed_covariance)


def test_fixed_observation_noise_is_jittable(
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
    koh_dataset_fixture: KOHDataset,
):
    X = koh_dataset_fixture.Xf

    kernel = minimal_koh_model_static_obs_fixture.k_epsilon(gpjax_params_fixture)
    expected = kernel.cross_covariance(X, X)
    actual = jit(lambda x: kernel.cross_covariance(x, x))(X)

    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize(
    ("model_class", "missing_method"),
    [
        (MissingEtaKOHModel, "k_eta"),
        (MissingDeltaKOHModel, "k_delta"),
    ],
)
def test_kohmodel_subclasses_must_implement_both_component_kernels(
    model_class, missing_method
):
    with pytest.raises(TypeError) as exc_info:
        model_class(None, None)

    # CPython formats this message differently across supported Python
    # versions (for example, "with abstract method" versus "without an
    # implementation for abstract method"). Check the stable information
    # instead of depending on the exact wording.
    message = str(exc_info.value)
    assert model_class.__name__ in message
    assert missing_method in message


def test_gp_kernel(
    minimal_koh_model_fixture: MinimalKOHModel,
    gpjax_params_fixture: dict,
    koh_dataset_fixture: KOHDataset,
):
    model = minimal_koh_model_fixture
    kernel = model.GP_kernel(gpjax_params_fixture)
    assert isinstance(kernel, KOHKernel)
    assert kernel.num_field_obs == koh_dataset_fixture.num_field_obs
    assert kernel.num_sim_obs == koh_dataset_fixture.num_sim_obs

    # Check sub-kernels (basic type checks, details depend on MinimalKOHModel's impl)
    assert isinstance(kernel.k_eta, gpx.kernels.RBF)
    assert isinstance(kernel.k_delta, gpx.kernels.RBF)
    assert isinstance(kernel.k_epsilon, gpx.kernels.White)
    assert isinstance(kernel.k_epsilon_eta, gpx.kernels.White)
    assert jnp.isclose(
        kernel.k_epsilon_eta.variance.value, 0.0
    )  # k_epsilon_eta is White(0.0)
    assert not isinstance(kernel.k_epsilon_eta.variance, Parameter)


def test_epsilon_eta_can_be_enabled_as_a_trainable_nonzero_parameter(
    minimal_koh_model_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
):
    params_with_epsilon_eta = {
        **gpjax_params_fixture,
        "epsilon_eta": {"variances": {"obs_noise": jnp.array(0.25)}},
    }

    kernel = minimal_koh_model_fixture.GP_kernel(params_with_epsilon_eta)

    assert jnp.isclose(kernel.k_epsilon_eta.variance.value, 0.25)
    assert isinstance(kernel.k_epsilon_eta.variance, Parameter)


def test_enabled_epsilon_eta_is_included_in_gpjax_trainable_state(
    minimal_koh_model_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
):
    params_with_epsilon_eta = {
        **gpjax_params_fixture,
        "epsilon_eta": {"variances": {"obs_noise": jnp.array(0.25)}},
    }
    posterior = minimal_koh_model_fixture.GP_posterior(params_with_epsilon_eta)
    _, trainable_state, *_ = nnx.split(posterior, Parameter, ...)
    trainable_paths = {path for path, _ in trainable_state.flat_state()}

    assert any(
        path[-2:] == ("k_epsilon_eta", "variance") for path in trainable_paths
    )


def test_likelihood(
    minimal_koh_model_fixture: MinimalKOHModel,
    gpjax_params_fixture: dict,
    koh_dataset_fixture: KOHDataset,
):
    model = minimal_koh_model_fixture
    num_total_obs = koh_dataset_fixture.num_field_obs + koh_dataset_fixture.num_sim_obs
    likelihood = model.likelihood(num_total_obs, gpjax_params_fixture)

    assert isinstance(likelihood, gpx.likelihoods.Gaussian)
    assert likelihood.num_datapoints == num_total_obs
    # obs_stddev is 0.0 because variance is handled in k_epsilon
    assert jnp.isclose(likelihood.obs_stddev.value, 0.0)
    assert not isinstance(likelihood.obs_stddev, Parameter)


def test_gp_prior(
    minimal_koh_model_fixture: MinimalKOHModel, gpjax_params_fixture: dict
):
    model = minimal_koh_model_fixture
    mean_func = model.GP_prior_mean_function()
    kernel = model.GP_kernel(gpjax_params_fixture)
    prior = model.GP_prior(mean_func, kernel)

    assert isinstance(prior, gpx.gps.Prior)
    assert prior.mean_function == mean_func
    assert prior.kernel == kernel
    assert prior.jitter == model.jitter


def test_gp_posterior(
    minimal_koh_model_fixture: MinimalKOHModel, gpjax_params_fixture: dict
):
    model = minimal_koh_model_fixture
    posterior = model.GP_posterior(gpjax_params_fixture)
    assert isinstance(
        posterior, KOHPosterior
    )  # or gpx.gps.Posterior if KOHPosterior is a direct alias/subclass

    # Check if prior and likelihood are of expected types
    assert isinstance(posterior.prior, gpx.gps.Prior)
    assert isinstance(posterior.likelihood, gpx.likelihoods.Gaussian)
    # Check if the kernel within the prior is a KOHKernel
    assert isinstance(posterior.prior.kernel, KOHKernel)


def test_fixed_observation_noise_is_excluded_from_gpjax_trainable_state(
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
):
    posterior = minimal_koh_model_static_obs_fixture.GP_posterior(
        gpjax_params_fixture
    )
    _, trainable_state, *_ = nnx.split(posterior, Parameter, ...)
    trainable_paths = {path for path, _ in trainable_state.flat_state()}

    assert not any(
        path[-2:] == ("k_epsilon", "variance") for path in trainable_paths
    )
    assert not any(
        path[-2:] == ("k_epsilon_eta", "variance") for path in trainable_paths
    )
    assert not any(path[-2:] == ("likelihood", "obs_stddev") for path in trainable_paths)

    # Other model hyperparameters remain trainable.
    assert any(path[-2:] == ("k_eta", "lengthscale") for path in trainable_paths)


def test_learned_observation_noise_is_included_in_gpjax_trainable_state(
    minimal_koh_model_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
):
    posterior = minimal_koh_model_fixture.GP_posterior(gpjax_params_fixture)
    _, trainable_state, *_ = nnx.split(posterior, Parameter, ...)
    trainable_paths = {path for path, _ in trainable_state.flat_state()}

    assert any(
        path[-2:] == ("k_epsilon", "variance") for path in trainable_paths
    )


def test_gpjax_fit_scipy_keeps_fixed_observation_noise_fixed(
    minimal_koh_model_static_obs_fixture: MinimalKOHModel,
    gpjax_params_fixture: ModelParameterDict,
    koh_dataset_fixture: KOHDataset,
):
    posterior = minimal_koh_model_static_obs_fixture.GP_posterior(
        gpjax_params_fixture
    )
    train_data = koh_dataset_fixture.get_dataset(jnp.array([[0.0]]))
    initial_fixed_variance = posterior.prior.kernel.k_epsilon.variance.value

    fitted_posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda model, data: -gpx.objectives.conjugate_mll(model, data),
        train_data=train_data,
        max_iters=2,
        verbose=False,
    )

    fitted_fixed_variance = fitted_posterior.prior.kernel.k_epsilon.variance.value
    assert jnp.array_equal(fitted_fixed_variance, initial_fixed_variance)


def test_get_koh_neg_log_pos_dens_func(
    minimal_koh_model_fixture: MinimalKOHModel,
    model_parameters_fixture: ModelParameters,
):
    model = minimal_koh_model_fixture
    neg_log_post_fn = model.get_KOH_neg_log_pos_dens_func()

    # Prepare a sample of unconstrained flat parameters
    # From simple_prior_dict: 1 (theta_0) + 2 (eta) + 2 (delta) + 1 (epsilon) = 6 params
    num_params = model_parameters_fixture.n_params
    mcmc_sample_flat = jnp.zeros(num_params)  # Using zeros for simplicity

    value = neg_log_post_fn(mcmc_sample_flat)
    assert isinstance(value, jnp.ndarray)
    assert value.shape == ()  # Scalar

    # Test JIT compilation
    jitted_fn = jit(neg_log_post_fn)
    value_jit = jitted_fn(mcmc_sample_flat)
    assert isinstance(value_jit, jnp.ndarray)
    assert value_jit.shape == ()
    assert jnp.allclose(value, value_jit)

    # A very simple value check would be extremely complex to set up manually.
    # The main test here is that it runs, JITs, and returns the correct type/shape.
    # And that checkify wrapper doesn't break it.
    # A NaN return would indicate serious issues.
    assert not jnp.isnan(value)


# End of test_kohmodel.py
