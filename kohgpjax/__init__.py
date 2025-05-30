from kohgpjax import (
    gps,
    kernels,
)
from kohgpjax.dataset import KOHDataset
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import (
    ModelParameters,
    ParameterPrior,
)

__description__ = (
    "Bayesian calibration from Kennedy & O'Hagan (2001) implementation in GPJax"
)
__url__ = "https://github.com/jamesbriant/KOH-GPJax"
__contributors__ = "James Briant - https://james.briant.co.uk"
__version__ = "0.4.0"

__all__ = [
    "gps",
    "kernels",
]
