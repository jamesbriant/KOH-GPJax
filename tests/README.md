# README.md

## Validation

The jupyter notebooks with filenames beginning with `valid-` are validation tests using data from a toy problem. They test a range of discrepancy functions and models to ensure the GPJax implementation is working as anticipated.

## Timings

Files beginning with `timings-` used the `matlabmodel` implementation to compare the GPJax code against the MATLAB implementation. The results of which can be found by running the `timings/elapsed_time_plots.py` script.

## Convergence

Convergence of the chains is analysed in files prefixed by `convergence-`.