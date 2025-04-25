import beartype.typing as tp
from beartype.typing import Tuple
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array
from jax.numpy import pad
from jax.scipy.linalg import block_diag
from jaxtyping import Float

K = tp.TypeVar("K", bound="AbstractKernel")

class KOHKernelComputation(AbstractKernelComputation):
    r"""Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def _calc_sub_kernels(
        self,
        kernel: K,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
    ) -> Tuple[Array, ...]:
        # PART 1 - Extract the field observations and simulation outputs
        a = kernel.num_field_obs
        b = kernel.num_sim_obs

        x_field = x[:a, ...]
        y_field = y[:a, ...]

        x_sim = x[a:a+b, ...]
        y_sim = y[a:a+b, ...]

        # PART 2 - Construct the cross-covariance sub-matrices
        sigma_eta = kernel.k_eta.cross_covariance(x, y)
        sigma_delta = kernel.k_delta.cross_covariance(x_field, y_field)
        sigma_epsilon_eta = kernel.k_epsilon_eta.cross_covariance(x_sim, y_sim)

        return sigma_eta, sigma_delta, sigma_epsilon_eta


    # def _cross_covariance( #TODO: Should this stay public rather than private? as cross_covariance()?
    def cross_covariance(
        self, 
        kernel: K,
        x: Float[Array, "N D"], 
        y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute the cross-covariance matrix.

        For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape $`NxD`$ and $`MxD`$.

        Args:
            kernel (Kernel): the kernel function.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed cross-covariance.
        """
        a = kernel.num_field_obs
        b = kernel.num_sim_obs
        N = x.shape[0]

        # PART 1 & 2
        sigma_eta, sigma_delta, sigma_epsilon_eta = self._calc_sub_kernels(
            kernel, x, y
        )

        # PART 3 - Construct the output array
        return sigma_eta + pad(
            block_diag(
                sigma_delta,
                sigma_epsilon_eta
            ), 
            (
                (0, N-a-b), 
                (0, N-a-b)
            )
        )