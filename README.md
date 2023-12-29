# KOH-GPJax

KOH-GPJax is an extension to the [GPJax](https://github.com/JaxGaussianProcesses/GPJax) Python package which implements the [Kennedy & O'Hagan (2001)](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00294)[^1] Bayesian Calibration of Computer Models framework.

By combining the power of [Jax](https://jax.readthedocs.io/en/latest/), the excellent modular design of GPJax and the latest gradient based MCMC methods, KOH-GPJax aims to provide a Bayesian calibration framework for large-scale computer simulations. Examples include nuclear fusion simulations (UKAEA) and weather simulations (UK Met Office).

[^1]: Kennedy, M.C. and O'Hagan, A. (2001), Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 425-464. https://doi.org/10.1111/1467-9868.00294

Very vague outline of this project: 
- [x] Create suitable kernel  
- [x] Build a posterior  
- [ ] Integrate with MICI
- [ ] Complete toy problem
- [ ] Complete harder problem (UKAEA)
- [ ] Complete very hard problem (UK Met Office)