from .._mod_replace import replace_modname
from ._base import BaseModule
from ._diff_preisach import DifferentiablePreisach, DifferentiablePreisachModel
from ._sa_preisach import SelfAdaptivePreisach, SelfAdaptivePreisachModel

for _mod in (
    SelfAdaptivePreisach,
    SelfAdaptivePreisachModel,
    DifferentiablePreisach,
    DifferentiablePreisachModel,
    BaseModule,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname


__all__ = [
    "BaseModule",
    "DifferentiablePreisach",
    "DifferentiablePreisachModel",
    "SelfAdaptivePreisach",
    "SelfAdaptivePreisachModel",
]
