from .._mod_replace import replace_modname
from ._binary_parameter import BinaryParameter
from ._constrained_parameter import ConstrainedParameter
from ._gpy_constrained_parameter import GPyConstrainedParameter

for _mod in (BinaryParameter, ConstrainedParameter, GPyConstrainedParameter):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "BinaryParameter",
    "ConstrainedParameter",
    "GPyConstrainedParameter",
]
