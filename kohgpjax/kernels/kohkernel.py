from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    # ScalarFloat,
)
from gpjax.kernels.computations import AbstractKernelComputation
# from gpjax.kernels.stationary import RBF
# from gpjax.parameters import Parameter, Static
from jaxtyping import Float

from kohgpjax.kernels.computations.kohcomputation import KOHKernelComputation


class KOHKernel(AbstractKernel):
    r"""Kennedy & O'Hagan (2001) kernel. Made up of subkernels which represent different parts of the data."""

    name: str = "KOHKernel"

    def __init__(
        self,
        num_field_obs: int,
        num_sim_obs: int,
        k_eta: AbstractKernel,
        k_delta: AbstractKernel,
        # k_epsilon: AbstractKernel,
        k_epsilon_eta: AbstractKernel,
    ) -> None:
        self.num_field_obs = num_field_obs
        self.num_sim_obs = num_sim_obs
        self.k_eta = k_eta
        self.k_delta = k_delta
        # self.k_epsilon = k_epsilon
        self.k_epsilon_eta = k_epsilon_eta

        self.compute_engine: AbstractKernelComputation = KOHKernelComputation()

    def __call__(
            self, 
            x: Float[Array, " D"], 
            y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        r"""
        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            Float: The value of $`k(x, y)`$.
        """
        raise NotImplementedError("It is no obvious how to compute the kernel value for this kernel. Instead calculate the desired value by calling one of the components (or subkernels) of this class.")