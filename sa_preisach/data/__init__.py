from .._mod_replace import replace_modname
from ._datamodule import PreisachDataModule

for _mod in (PreisachDataModule,):
    replace_modname(_mod, __name__)


del _mod
del replace_modname

__all__ = [
    "PreisachDataModule",
]
