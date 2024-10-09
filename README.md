# KOH-GPJax

## Introduction
KOH-GPJax is an extension to the [GPJax](https://github.com/JaxGaussianProcesses/GPJax) Python package which implements the [Kennedy & O'Hagan (2001)](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00294)[^1] Bayesian Calibration of Computer Models framework.

By combining the power of [Jax](https://jax.readthedocs.io/en/latest/), the excellent modular design of GPJax and the latest gradient based MCMC methods, KOH-GPJax aims to provide a Bayesian calibration framework for large-scale computer simulations. Examples include nuclear fusion simulations (UKAEA) and weather simulations (UK Met Office).

## Where to Start

Note: at the time of writing the latest version of GPJax (0.8.2) is not compatitable with python 3.12. Instead use python 3.11.

- MCMC schemes are implemented in jupyter notebooks starting with `mici_[...].ipynb`.
- Comparisons between Mici implementations and GPJax implementations are in jupyter notebooks titled `MATLAB_vs_Mici.ipynb` and similar.
- Model implementations are in files titled `matlabmodel.py` and similar.
- The files which extend GPJax are under the `kohgpjax/` directory.

## Installation

To run the notebooks under `tests/` install the kohgpjax package by executing `pip install -e .` in the root directory.

## To-Do
Very vague outline of this project: 
- [x] Create suitable kernel  
- [x] Build a posterior  
- [x] Integrate with MICI
- [x] Complete toy problem
- [ ] Complete harder problem (UKAEA)
- [ ] Complete very hard problem (UK Met Office)

## References
[^1]: Kennedy, M.C. and O'Hagan, A. (2001), Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 425-464. https://doi.org/10.1111/1467-9868.00294