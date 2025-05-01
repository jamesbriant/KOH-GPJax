# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

import gpjax as gpx
key = jr.key(123)



n = 100
noise = 0.3

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal #+ jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations")
ax.plot(xtest, ytest, label="Latent function")
ax.legend(loc="best")
plt.show()



kernel = gpx.kernels.RBF()  # 1-dimensional input
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

prior_dist = prior.predict(xtest)

prior_mean = prior_dist.mean
prior_std = prior_dist.variance
samples = prior_dist.sample(key=key, sample_shape=(20,))

fig, ax = plt.subplots()
ax.plot(xtest, samples.T, alpha=0.5)
ax.plot(xtest, prior_mean, label="Prior mean")
ax.fill_between(
    xtest.flatten(),
    prior_mean - prior_std,
    prior_mean + prior_std,
    alpha=0.3,
    label="Prior variance",
)
ax.legend(loc="best")
plt.show()



likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=gpx.parameters.Static(0.0))
posterior = prior * likelihood

print(-gpx.objectives.conjugate_mll(posterior, D))

opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    # we use the negative mll as we are minimising
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
)

print(-gpx.objectives.conjugate_mll(opt_posterior, D))
print("obs_stddev:", opt_posterior.likelihood.obs_stddev.value)


latent_dist = opt_posterior.predict(xtest, train_data=D)
predictive_dist = opt_posterior.likelihood(latent_dist)

predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)


fig, ax = plt.subplots(figsize=(7.5, 2.5))
ax.plot(x, y, "x", label="Observations", alpha=0.5)
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    alpha=0.2,
    label="Two sigma",
)
ax.plot(
    xtest,
    predictive_mean - 2 * predictive_std,
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + 2 * predictive_std,
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest, ytest, label="Latent function", linestyle="--", linewidth=2
)
ax.plot(xtest, predictive_mean, label="Predictive mean")
ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
plt.show()