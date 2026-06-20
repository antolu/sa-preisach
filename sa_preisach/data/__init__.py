from .._mod_replace import replace_modname
from ._cycled_datamodule import CycleBatch, CycleDataset, CycledPreisachDataModule, cycle_collate_fn
from ._datamodule import PreisachDataModule

for _mod in (CycledPreisachDataModule, PreisachDataModule):
    replace_modname(_mod, __name__)


del _mod
del replace_modname

__all__ = [
    "CycleBatch",
    "CycleDataset",
    "CycledPreisachDataModule",
    "PreisachDataModule",
    "cycle_collate_fn",
]
