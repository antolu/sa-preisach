from .._mod_replace import replace_modname
from ._datamodule import PreisachDataModule
from ._encoder_decoder_datamodule import EncoderDecoderPreisachDataModule

for _mod in (PreisachDataModule, EncoderDecoderPreisachDataModule):
    replace_modname(_mod, __name__)


del _mod
del replace_modname

__all__ = [
    "EncoderDecoderPreisachDataModule",
    "PreisachDataModule",
]
