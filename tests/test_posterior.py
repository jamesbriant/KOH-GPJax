from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import (
    jit,
    numpy as jnp,
)
import numpy as np
import numpyro.distributions as npd

from kohgpjax.dataset import KOHDataset
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import (
    ModelParameters,
    ParameterPrior,
    PriorDict,
)

DATAFIELD = np.loadtxt(
    "/Users/jamesbriant/Documents/Projects/BayesianCalibrationExamples/presentable/data/toy/field.csv",
    delimiter=",",
    dtype=np.float32,
)
DATACOMP = np.loadtxt(
    "/Users/jamesbriant/Documents/Projects/BayesianCalibrationExamples/presentable/data/toy/sim.csv",
    delimiter=",",
    dtype=np.float32,
)

xf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
xc = jnp.reshape(DATACOMP[:, 0], (-1, 1)).astype(jnp.float64)
tc = jnp.reshape(DATACOMP[:, 1], (-1, 1)).astype(jnp.float64)
yf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
yc = jnp.reshape(DATACOMP[:, 2], (-1, 1)).astype(jnp.float64)

field_dataset = gpx.Dataset(xf, yf)
comp_dataset = gpx.Dataset(jnp.hstack((xc, tc)), yc)

kohdataset = KOHDataset(field_dataset, comp_dataset)

prior_dict: PriorDict = {
    "thetas": {
        "theta_0": ParameterPrior(
            npd.Normal(loc=0.0, scale=1.0),
        ),
    },
    "eta": {  # TODO: Change names to 'emulator', 'discrepancy', 'obs_noise', '???'?
        "variance": ParameterPrior(
            npd.Normal(loc=0.0, scale=0.5),
        ),
        "lengthscale": {
            "x_0": ParameterPrior(
                npd.Normal(loc=0.0, scale=1.0),
            ),
            "theta_0": ParameterPrior(
                npd.Normal(loc=0.0, scale=1.0),
            ),
        },
    },
    "delta": {
        "variance": ParameterPrior(
            npd.Normal(loc=0.0, scale=0.5),
        ),
        "lengthscale": {
            "x_0": ParameterPrior(
                npd.Normal(loc=0.0, scale=1.0),
            ),
        },
    },
    "epsilon": {
        "variance": ParameterPrior(
            npd.Normal(loc=0.0, scale=0.5),
        ),
    },
    "epsilon_eta": {  # TODO: make this optional
        "variance": ParameterPrior(
            npd.Normal(loc=0.1, scale=0.5),
        ),
    },
}


class MyModel(KOHModel):
    def k_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.ProductKernel(
            kernels=[
                gpx.kernels.RBF(
                    active_dims=[0],
                    lengthscale=jnp.array(GPJAX_params["lengthscale"]["x_0"]),
                    variance=jnp.array(1 / GPJAX_params["variance"]),
                ),
                gpx.kernels.RBF(
                    active_dims=[1],
                    lengthscale=jnp.array(GPJAX_params["lengthscale"]["theta_0"]),
                ),
            ]
        )

    def k_delta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.RBF(
            active_dims=[0],
            lengthscale=jnp.array(GPJAX_params["lengthscale"]["x_0"]),
            variance=jnp.array(1 / GPJAX_params["variance"]),
        )

    def k_epsilon_eta(self, GPJAX_params) -> gpx.kernels.AbstractKernel:
        return gpx.kernels.White(
            active_dims=[0], variance=jnp.array(1 / GPJAX_params["variance"])
        )


model_params = ModelParameters(prior_dict)
model = MyModel(
    model_parameters=model_params,
    kohdataset=kohdataset,
)

f = jit(model.get_KOH_neg_log_pos_dens_func())
mcmc_params = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
print(f(mcmc_params))
# 707.3885600290141
