from jax import config

config.update("jax_enable_x64", True)

from gpjax.dataset import Dataset
from jax import numpy as jnp

from kohgpjax.dataset import KOHDataset

x_field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y_field = jnp.array([[1.0], [2.0]])
field_dataset = Dataset(x_field, y_field)

x_sim = jnp.array([[1.0, 2.0, 3.0, 2.0], [4.0, 5.0, 6.0, 3.0]])
y_sim = jnp.array([[1.0], [2.0]])
sim_dataset = Dataset(x_sim, y_sim)

koh_dataset = KOHDataset(field_dataset, sim_dataset)
print(koh_dataset)
# Expected output:
# KOHDataset(
#   Datasets:
#     Field data = Dataset(Number of observations: 2 - Input dimension: 2),
#     Simulation data = Dataset(Number of observations: 2 - Input dimension: 4)
#   Attributes:
#     No. field observations = 2,
#     No. simulation outputs = 2,
#     No. variable params = 2,
#     No. calibration params = 2,
# )

print(koh_dataset.d)
# Expected output:
# [[1.]
#  [2.]
#  [1.]
#  [2.]]

theta = jnp.array([1.5, 2.5])
# print(koh_dataset.X(theta))
# Expected output:
# ValueError: Parameter theta must be a 2D array. Got theta.ndim=1

theta = jnp.array([[1.5, 2.5]])
print(koh_dataset.X(theta))
# Expected output:
# [[1.  2.  1.5 2.5]
#  [3.  4.  1.5 2.5]
#  [1.  2.  3.  2. ]
#  [4.  5.  6.  3. ]]

theta = jnp.array([[1.5], [2.5]])
print(koh_dataset.X(theta))
# Expected output:
# [[1.  2.  1.5 2.5]
#  [3.  4.  1.5 2.5]
#  [1.  2.  3.  2. ]
#  [4.  5.  6.  3. ]]
