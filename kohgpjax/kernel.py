# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass

from jaxtyping import Float

from gpjax.base import param_field, static_field
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from kohgpjax.computation import KOHKernelComputation
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.kernels.stationary import RBF


@dataclass
class KOHKernel(AbstractKernel):
    r"""Kennedy & O'Hagan (2001) kernel. Made up of subkernels which represent different parts of the data."""

    num_field_obs: int = static_field(None)
    num_sim_obs: int = static_field(None)
    k_eta: AbstractKernel = param_field(RBF())
    k_delta: AbstractKernel = param_field(RBF())
    k_epsilon: AbstractKernel = param_field(RBF())
    k_epsilon_eta: AbstractKernel = param_field(RBF())
    # theta: param_field(jnp.array(0.0))
    name: str = static_field("KOHKernel")
    compute_engine: AbstractKernelComputation = static_field(
        KOHKernelComputation(), repr=False
    )
    # any attribute of a module is a trainable parameter unless it is a static_field()

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""
        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        # pass
        raise NotImplementedError