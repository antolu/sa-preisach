from .._mod_replace import replace_modname
from ._sa_preisach import SelfAdaptivePreisach, SelfAdaptivePreisachModel

for _mod in (SelfAdaptivePreisach, SelfAdaptivePreisachModel):
    replace_modname(_mod, __name__)

del _mod
del replace_modname


__all__ = [
    "SelfAdaptivePreisach",
    "SelfAdaptivePreisachModel",
]
