from dataclasses import dataclass

from gpjax.dataset import Dataset
from gpjax.typing import Array
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Num

@dataclass
@register_pytree_node_class
class KOHDataset:
    r"""
    A class to handle the simulation and observation data for the KOH framework.

    Args:
        field_dataset: The observation data of size $M \times P$.
        sim_dataset: The simulation data of size $N \times P+Q$. The first $P$ columns
            are the input data, and the last $Q$ columns are the calibration parameters.
    """

    field_dataset: Dataset
    sim_dataset: Dataset

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $z$, $y$, $X_f$ and $X_s$ are compatible"""
        _check_shapes(self.sim_dataset, self.field_dataset)

        self.num_sim_obs = self.sim_dataset.y.shape[0]
        self.num_field_obs = self.field_dataset.y.shape[0]
        self.num_calib_params = self.sim_dataset.X.shape[1] - self.field_dataset.X.shape[1]

    def __repr__(self) -> str:
        r"""Returns a string representation of the KOHDataset instance."""
        repr = (
            f"KOHDataset(\n"
            f"  Datasets:\n"
            f"    Field data = {self.field_dataset},\n"
            f"    Simulation data = {self.sim_dataset}\n"
            f"  Attributes:\n"
            f"    No. field observations = {self.num_field_obs},\n"
            f"    No. simulation outputs = {self.num_sim_obs},\n"
            f"    No. variable params = {self.field_dataset.X.shape[1]},\n"
            f"    No. calibration params = {self.num_calib_params},\n"
            f")"
        )
        return repr
    
    @property
    def z(self) -> Num[Array, "n 1"]:
        r"""Returns the field observations."""
        return self.field_dataset.y
    
    @property
    def y(self) -> Num[Array, "N 1"]:
        r"""Returns the simulation output."""
        return self.sim_dataset.y
    
    @property
    def d(self) -> Num[Array, "n+N 1"]:
        r"""Returns the field observations stacked above the simulation output.
        Note this is the opposite of the KOH paper. For no good reason really.
        Should this be changed? What are the numerical implications?"""
        return jnp.vstack((self.z, self.y))
    
    @property
    def Xf(self) -> Num[Array, "n P"]:
        r"""Returns the input data for the field observations."""
        return self.field_dataset.X
    
    # This is NOT a property because it takes an argument.
    def Xf_theta(self, theta: Num[Array, "1 Q"]) -> Num[Array, "n P+Q"]:
        r"""Returns the input data for the field observations and calibration parameters."""
        _check_theta_shape(theta, self.num_calib_params)
        theta = theta.reshape(1, -1)
        theta = jnp.repeat(theta, self.num_field_obs, axis=0)
        return jnp.hstack((self.Xf, theta))
    
    @property
    def Xc(self) -> Num[Array, "N P+Q"]:
        r"""Returns the input data for the simulation output."""
        return self.sim_dataset.X

    # This is NOT a property because it takes an argument.
    def X(self, theta: Num[Array, "1 Q"]) -> Num[Array, "n+N P+Q"]:
        r"""Returns the input data for the field observations and simulation output."""
        return jnp.vstack((self.Xf_theta(theta), self.Xc))
    
    def get_dataset(self, theta: Num[Array, "1 Q"]) -> Dataset:
        r"""Returns the dataset with the field observations and the simulator observations
        concatenated with the given theta."""
        return Dataset(X=self.X(theta), y=self.d)
    
    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.d.shape[0]
    
    def tree_flatten(self):
        return (self.field_dataset, self.sim_dataset), None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _check_shapes(
    sim_dataset: Dataset,
    field_dataset: Dataset
) -> None:
    r"""Check that the shapes of the simulation and observation datasets are compatible."""
    if sim_dataset.X.shape[1] <= field_dataset.X.shape[1]:
        raise ValueError(
            f"Input dimension of simulation data ({sim_dataset.X.shape[1]}) "
            f"must be greater than input dimension of field data ({field_dataset.X.shape[1]})"
        )
    
    if field_dataset.y.ndim > 2 or field_dataset.y.shape[1] != 1:
        raise ValueError(
            f"Field observations must have shape (n, 1). Got shape={field_dataset.y.shape}."
        )
    
    if sim_dataset.y.ndim > 2 or sim_dataset.y.shape[1] != 1:
        raise ValueError(
            f"Simulation outputs must have shape (N, 1). Got shape={sim_dataset.y.shape}."
        )    


def _check_theta_shape(theta: Array, num_calib_params: int) -> None:
    r"""Check that the shape of the calibration parameters is correct."""
    if theta.ndim != 2:
        raise ValueError(
            f"Parameter theta must be a 2D array. Got theta.ndim={theta.ndim}"
        )
    
    if theta.shape != (num_calib_params, 1) and theta.shape != (1, num_calib_params):
        raise ValueError(
            f"Parameter theta must have shape (Q, 1) OR (1, Q). Got theta.shape={theta.shape}"
        )