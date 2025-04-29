from .._mod_replace import replace_modname
from ._plot import PlotHysteresisCallback

replace_modname(PlotHysteresisCallback, __name__)

del replace_modname

__all__ = [
    "PlotHysteresisCallback",
]
