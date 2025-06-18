from gpjax.dataset import Dataset
from jax import numpy as jnp  # Removed 'config'
import pytest

from kohgpjax.dataset import KOHDataset

# --- Fixtures ---


@pytest.fixture(scope="module")
def field_data_input():
    x_field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y_field = jnp.array([[1.0], [2.0]])
    return Dataset(x_field, y_field)


@pytest.fixture(scope="module")
def sim_data_input():
    # x_sim has 2 variable params and 2 calibration params
    x_sim = jnp.array([[1.0, 2.0, 3.0, 2.0], [4.0, 5.0, 6.0, 3.0]])
    y_sim = jnp.array([[1.0], [2.0]])
    return Dataset(x_sim, y_sim)


@pytest.fixture
def koh_dataset_instance(field_data_input, sim_data_input):
    return KOHDataset(field_data_input, sim_data_input)


# --- Tests ---


def test_koh_dataset_attributes(koh_dataset_instance, field_data_input, sim_data_input):
    """Tests the initialization and basic attributes of KOHDataset."""
    assert koh_dataset_instance.field_dataset == field_data_input
    assert koh_dataset_instance.sim_dataset == sim_data_input
    assert koh_dataset_instance.num_field_obs == 2
    assert koh_dataset_instance.num_sim_obs == 2
    assert koh_dataset_instance.num_variable_params == 2  # x_field.shape[1]
    assert (
        koh_dataset_instance.num_calib_params == 2
    )  # x_sim.shape[1] - x_field.shape[1]


def test_koh_dataset_d_property(koh_dataset_instance):
    """Tests the 'd' property of KOHDataset."""
    expected_d = jnp.array([[1.0], [2.0], [1.0], [2.0]])  # field_y then sim_y
    assert jnp.array_equal(koh_dataset_instance.d, expected_d)


def test_koh_dataset_X_method_valid_input_1_2_theta(koh_dataset_instance):
    """Tests the 'X' method with theta of shape (1, num_calib_params)."""
    theta = jnp.array([[1.5, 2.5]])  # Shape (1, 2)

    expected_X_array = jnp.array(
        [
            [1.0, 2.0, 1.5, 2.5],
            [3.0, 4.0, 1.5, 2.5],
            [1.0, 2.0, 3.0, 2.0],
            [4.0, 5.0, 6.0, 3.0],
        ]
    )
    assert jnp.array_equal(koh_dataset_instance.X(theta), expected_X_array)


def test_koh_dataset_X_method_valid_input_2_1_theta(koh_dataset_instance):
    """Tests the 'X' method with theta of shape (num_calib_params, 1)."""
    theta = jnp.array([[1.5], [2.5]])  # Shape (2, 1) num_calib_params=2

    expected_X_array = jnp.array(
        [
            [1.0, 2.0, 1.5, 2.5],
            [3.0, 4.0, 1.5, 2.5],
            [1.0, 2.0, 3.0, 2.0],
            [4.0, 5.0, 6.0, 3.0],
        ]
    )
    # The Xf_theta method reshapes theta to (1, num_calib_params)
    # So jnp.array([[1.5], [2.5]]) which is (2,1) becomes jnp.array([[1.5, 2.5]]) which is (1,2)
    # when num_calib_params is 2.
    assert jnp.array_equal(koh_dataset_instance.X(theta), expected_X_array)


def test_koh_dataset_X_method_invalid_theta_ndim(koh_dataset_instance):
    """Tests the 'X' method for ValueError when theta.ndim=1."""
    theta_invalid_ndim = jnp.array([1.5, 2.5])  # Shape (2,) which is ndim=1
    with pytest.raises(
        ValueError, match="Parameter theta must be a 2D array. Got theta.ndim=1"
    ):
        koh_dataset_instance.X(theta_invalid_ndim)


def test_koh_dataset_X_method_invalid_theta_shape(koh_dataset_instance):
    """Tests the 'X' method for ValueError when theta has an incompatible 2D shape."""
    # num_calib_params is 2 for koh_dataset_instance.
    # This theta is (1,3)
    theta_invalid_shape_1_3 = jnp.array([[1.5, 2.5, 3.5]])
    with pytest.raises(
        ValueError,
        match=r"Parameter theta must have shape \(2, 1\) OR \(1, 2\). Got theta.shape=\(1, 3\)",
    ):
        koh_dataset_instance.X(theta_invalid_shape_1_3)

    # This theta is (3,1)
    theta_invalid_shape_3_1 = jnp.array([[1.5], [2.5], [3.5]])
    with pytest.raises(
        ValueError,
        match=r"Parameter theta must have shape \(2, 1\) OR \(1, 2\). Got theta.shape=\(3, 1\)",
    ):
        koh_dataset_instance.X(theta_invalid_shape_3_1)

    # This theta is (2,2) but num_calib_params is 2. This should be valid if it was (1,2) or (2,1)
    # but (2,2) is not (Q,1) or (1,Q) if Q=2.
    # _check_theta_shape has `if theta.shape not in ((num_calib_params, 1), (1, num_calib_params)):`
    # So for num_calib_params = 2, allowed shapes are (2,1) and (1,2). (2,2) is not allowed.
    theta_invalid_shape_2_2 = jnp.array([[1.5, 2.5], [3.5, 4.5]])
    with pytest.raises(
        ValueError,
        match=r"Parameter theta must have shape \(2, 1\) OR \(1, 2\). Got theta.shape=\(2, 2\)",
    ):
        koh_dataset_instance.X(theta_invalid_shape_2_2)


def test_koh_dataset_init_incompatible_sim_field_dimensions():
    """
    Tests KOHDataset initialization with simulation data input dimension
    not greater than field data input dimension.
    """
    x_field_ko = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y_field_ko = jnp.array([[1.0], [2.0]])
    field_dataset_ko = Dataset(x_field_ko, y_field_ko)

    x_sim_ko_equal_dim = jnp.array([[1.0, 2.0], [4.0, 5.0]])
    y_sim_ko = jnp.array([[1.0], [2.0]])
    sim_dataset_ko_equal_dim = Dataset(x_sim_ko_equal_dim, y_sim_ko)

    with pytest.raises(
        ValueError,
        match=r"Input dimension of simulation data \(2\) must be greater than input dimension of field data \(2\)",
    ):
        KOHDataset(field_dataset_ko, sim_dataset_ko_equal_dim)

    x_sim_ko_less_dim = jnp.array([[1.0], [4.0]])
    sim_dataset_ko_less_dim = Dataset(x_sim_ko_less_dim, y_sim_ko)
    with pytest.raises(
        ValueError,
        match=r"Input dimension of simulation data \(1\) must be greater than input dimension of field data \(2\)",
    ):
        KOHDataset(field_dataset_ko, sim_dataset_ko_less_dim)


def test_koh_dataset_init_invalid_y_field_shape(
    sim_data_input,
):  # sim_data_input is valid and can be reused
    """Tests KOHDataset initialization with invalid field_dataset.y shape."""
    # x_field_ko must have same number of rows as y_field_ko_shape_n2
    x_field_ko = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2,2)

    # Invalid y_field: ndim=1 - This is caught by gpjax.Dataset, so KOHDataset's check isn't reached.
    # We will test the shape[1] != 1 part of KOHDataset's _check_shapes function.
    # y_field_ko_ndim1 = jnp.array([1.0])
    # field_dataset_ko_ndim1 = Dataset(x_field_ko, y_field_ko_ndim1) # This line would fail
    # with pytest.raises(ValueError, match=r"Field observations must have shape \(n, 1\). Got shape=\(1,\)."):
    #      KOHDataset(field_dataset_ko_ndim1, sim_data_input)

    # Invalid y_field: shape (n,2) instead of (n,1)
    y_field_ko_shape_n2 = jnp.array([[1.0, 0.5], [2.0, 0.5]])  # Shape (2,2)
    field_dataset_ko_shape_n2 = Dataset(
        x_field_ko, y_field_ko_shape_n2
    )  # Now x_field_ko (2,2) and y_field_ko_shape_n2 (2,2) match rows
    # KOHDataset's _check_shapes raises:
    # "Field observations must have shape (n, 1). Got shape=(2, 2)." for field_dataset.y.shape[1] != 1
    with pytest.raises(
        ValueError,
        match=r"Field observations must have shape \(n, 1\). Got shape=\(2, 2\).",
    ):
        KOHDataset(field_dataset_ko_shape_n2, sim_data_input)


def test_koh_dataset_init_invalid_y_sim_shape(
    field_data_input,
):  # field_data_input is valid and can be reused
    """Tests KOHDataset initialization with invalid sim_dataset.y shape."""
    # field_data_input.X has shape (2,2)
    # x_sim_ko must have same num_rows as y_sim_ko_shape_n2, and more cols than field_data_input.X
    x_sim_ko = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2,3)

    # Invalid y_sim: ndim=1 - Caught by gpjax.Dataset
    # y_sim_ko_ndim1 = jnp.array([1.0])
    # sim_dataset_ko_ndim1 = Dataset(x_sim_ko, y_sim_ko_ndim1) # This line would fail
    # with pytest.raises(ValueError, match=r"Simulation outputs must have shape \(N, 1\). Got shape=\(1,\)."):
    #      KOHDataset(field_data_input, sim_dataset_ko_ndim1)

    # Invalid y_sim: shape (N,2) instead of (N,1)
    y_sim_ko_shape_n2 = jnp.array([[1.0, 0.5], [2.0, 0.5]])  # Shape (2,2)
    sim_dataset_ko_shape_n2 = Dataset(
        x_sim_ko, y_sim_ko_shape_n2
    )  # x_sim_ko (2,3) and y_sim_ko_shape_n2 (2,2) match rows
    # KOHDataset's _check_shapes raises:
    # "Simulation outputs must have shape (N, 1). Got shape=(2, 2)." for sim_dataset.y.shape[1] != 1
    with pytest.raises(
        ValueError,
        match=r"Simulation outputs must have shape \(N, 1\). Got shape=\(2, 2\).",
    ):
        KOHDataset(field_data_input, sim_dataset_ko_shape_n2)


def test_koh_dataset_repr(koh_dataset_instance):
    """Tests the __repr__ method of KOHDataset."""
    representation = repr(koh_dataset_instance)
    assert "KOHDataset(" in representation
    assert (
        "Field data = Dataset(Number of observations: 2 - Input dimension: 2)"
        in representation
    )
    assert (
        "Simulation data = Dataset(Number of observations: 2 - Input dimension: 4)"
        in representation
    )
    assert "No. field observations = 2" in representation
    assert "No. simulation outputs = 2" in representation
    assert "No. variable params = 2" in representation
    assert "No. calibration params = 2" in representation


def test_koh_dataset_z_property(koh_dataset_instance, field_data_input):
    """Tests the 'z' (field_dataset.y) property."""
    assert jnp.array_equal(koh_dataset_instance.z, field_data_input.y)


def test_koh_dataset_y_property(koh_dataset_instance, sim_data_input):
    """Tests the 'y' (sim_dataset.y) property."""
    assert jnp.array_equal(koh_dataset_instance.y, sim_data_input.y)


def test_koh_dataset_Xf_property(koh_dataset_instance, field_data_input):
    """Tests the 'Xf' (field_dataset.X) property."""
    assert jnp.array_equal(koh_dataset_instance.Xf, field_data_input.X)


def test_koh_dataset_Xc_property(koh_dataset_instance, sim_data_input):
    """Tests the 'Xc' (sim_dataset.X) property."""
    assert jnp.array_equal(koh_dataset_instance.Xc, sim_data_input.X)


def test_koh_dataset_Xf_theta_method(koh_dataset_instance):
    """Tests the Xf_theta method."""
    theta = jnp.array([[1.5, 2.5]])  # num_calib_params = 2
    expected_Xf_theta = jnp.array([[1.0, 2.0, 1.5, 2.5], [3.0, 4.0, 1.5, 2.5]])
    assert jnp.array_equal(koh_dataset_instance.Xf_theta(theta), expected_Xf_theta)

    # Test with theta that needs reshaping (Q,1) -> (1,Q)
    theta_q_1 = jnp.array([[1.5], [2.5]])
    assert jnp.array_equal(koh_dataset_instance.Xf_theta(theta_q_1), expected_Xf_theta)


def test_koh_dataset_get_dataset_method(koh_dataset_instance):
    """Tests the get_dataset method."""
    theta = jnp.array([[1.5, 2.5]])
    ds = koh_dataset_instance.get_dataset(theta)

    expected_X = koh_dataset_instance.X(theta)
    expected_y = koh_dataset_instance.d

    assert isinstance(ds, Dataset)
    assert jnp.array_equal(ds.X, expected_X)
    assert jnp.array_equal(ds.y, expected_y)


def test_koh_dataset_n_property(koh_dataset_instance):
    """Tests the 'n' property (total number of observations)."""
    # num_field_obs = 2, num_sim_obs = 2. So, n = 4
    assert koh_dataset_instance.n == 4


# Test tree_flatten and tree_unflatten
def test_koh_dataset_tree_flatten_unflatten(
    koh_dataset_instance, field_data_input, sim_data_input
):
    """Tests the tree_flatten and tree_unflatten methods for PyTree registration."""
    children, aux_data = koh_dataset_instance.tree_flatten()

    assert len(children) == 2
    assert children[0] == field_data_input
    assert children[1] == sim_data_input
    assert aux_data is None  # As per current implementation

    new_instance = KOHDataset.tree_unflatten(aux_data, children)
    assert isinstance(new_instance, KOHDataset)
    assert new_instance.field_dataset == field_data_input
    assert new_instance.sim_dataset == sim_data_input
    assert (
        new_instance.num_calib_params == koh_dataset_instance.num_calib_params
    )  # Check one derived attribute
    assert jnp.array_equal(new_instance.d, koh_dataset_instance.d)  # Check a property
