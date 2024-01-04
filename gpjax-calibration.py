# Enable Float64 for more stable matrix inversions.
from jax import config
config.update("jax_enable_x64", True)

import numpy as np

import jax.numpy as jnp
from jax import random
from jax import jit

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

import gpjax as gpx
from kohgpjax.kohkernel import KOHKernel

import matplotlib.pyplot as plt

key = random.PRNGKey(123)

################
##### DATA #####
################

DATAFIELD = np.loadtxt('data/simple_field.csv', delimiter=',', dtype=np.float32)
DATACOMP = np.loadtxt('data/simple_comp.csv', delimiter=',', dtype=np.float32)

num_obs = DATAFIELD.shape[0]

xf = np.reshape(DATAFIELD[:, 0], (-1, 1))
xc = np.reshape(DATACOMP[:, 0], (-1,1))
tc = np.reshape(DATACOMP[:, 1], (-1,1))
yf = np.reshape(DATAFIELD[:, 1], (-1,1))
yc = np.reshape(DATACOMP[:, 2], (-1,1))



#Standardize full response using mean and std of yc
yc_mean = np.mean(yc)
# yc_std = np.std(yc)
yc_std = np.std(yc, ddof=1) #estimate is now unbiased
x_min = min(xf.min(), xc.min())
x_max = max(xf.max(), xc.max())
t_min = tc.min()
t_max = tc.max()

xf_normalized = (xf - x_min)/(x_max - x_min)
xc_normalized = (xc - x_min)/(x_max - x_min)
# tc_normalized = np.zeros_like(tc)
# for k in range(tc.shape[1]):
#     tc_normalized[:, k] = (tc[:, k] - np.min(tc[:, k]))/(np.max(tc[:, k]) - np.min(tc[:, k]))
tc_normalized = (tc - t_min)/(t_max - t_min)
yc_standardized = (yc - yc_mean)/yc_std
yf_standardized = (yf - yc_mean)/yc_std

theta = 0.5

x = jnp.vstack((xf_normalized, xc_normalized))
t = jnp.vstack((jnp.zeros((xf_normalized.shape[0], tc_normalized.shape[1])) + theta, tc_normalized))
x = jnp.hstack((x, t))
y = jnp.vstack((yf_standardized, yc_standardized))

data = gpx.Dataset(X=x, y=y)

#################
##### PRIOR #####
#################

product_kernel = gpx.kernels.ProductKernel(kernels=[
        gpx.kernels.RBF(
            active_dims=[0],
            lengthscale=jnp.array(1/jnp.sqrt(2*50)),
        ), 
        gpx.kernels.RBF(
            active_dims=[1],
            lengthscale=jnp.array(1/jnp.sqrt(2*7)),
        )
    ])

kernel = KOHKernel(
    num_obs=num_obs,
    k_eta=product_kernel,
    k_delta=gpx.kernels.White(
        active_dims=[0],
        # lengthscale=jnp.array(1/jnp.sqrt(2*2)),
        variance=jnp.array(1/30)
    ), 
    k_epsilon=gpx.kernels.White(
        active_dims=[0],
        variance=jnp.array(1/1000)
    ),
    k_epsilon_eta=gpx.kernels.White(
        active_dims=[0],
        variance=jnp.array(1/10000)
    ),
)
meanf = gpx.mean_functions.Zero()
prior = gpx.Prior(
    mean_function=meanf, 
    kernel=kernel,
    jitter=0.
)

########################
##### SAMPLE PRIOR #####
########################

# num_points = 100

# xtest = jnp.linspace(0, 10, num_points).reshape(-1, 1)

# # ttest = jnp.linspace(0.2, 0.8, num_points).reshape(-1,1)
# ttest = jnp.array([0.4]*num_points).reshape(-1, 1)

# xtest = jnp.hstack((xtest, ttest))

# prior_dist = prior.predict(xtest)
# print(prior_dist)

# prior_mean = prior_dist.mean()
# prior_std = prior_dist.variance()
# samples = prior_dist.sample(seed=key, sample_shape=(5,))

# fig, ax = plt.subplots()
# # ax.plot(xtest[:,0], samples.T, alpha=0.5, label="Prior samples")
# ax.plot(xtest[:,0], prior_mean, label="Prior mean")
# ax.fill_between(
#     xtest[:,0].flatten(),
#     prior_mean - prior_std,
#     prior_mean + prior_std,
#     alpha=0.3,
#     label="Prior variance",
# )

#####################
##### POSTERIOR #####
#####################

likelihood = gpx.likelihoods.Gaussian(
    num_datapoints=data.n,
    obs_stddev=jnp.array(0.0)
)

posterior = prior * likelihood

negative_mll = gpx.objectives.ConjugateMLL(negative=True)
nll = negative_mll(posterior, train_data=data)
print(nll - 0.5*data.n*jnp.log(2*jnp.pi))