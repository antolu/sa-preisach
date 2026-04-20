from .._mod_replace import replace_modname
from ._base import DensityPrior
from ._composite import CompositeDensityPrior
from ._diagonal import DiagonalDensityPrior
from ._symmetry import SymmetryDensityPrior

replace_modname(DensityPrior, __name__)
replace_modname(CompositeDensityPrior, __name__)
replace_modname(DiagonalDensityPrior, __name__)
replace_modname(SymmetryDensityPrior, __name__)

del replace_modname

__all__ = [
    "CompositeDensityPrior",
    "DensityPrior",
    "DiagonalDensityPrior",
    "SymmetryDensityPrior",
]
