from .._mod_replace import replace_modname
from ._base import BaseModule
from ._diff_preisach import DifferentiablePreisach, DifferentiablePreisachModel
from ._diff_preisach_nn import DifferentiablePreisachNN, DifferentiablePreisachNNModel
from ._encoder_decoder_preisach_nn import (
    EncoderDecoderPreisachNN,
    EncoderDecoderPreisachNNModel,
)
from ._sa_preisach import SelfAdaptivePreisach, SelfAdaptivePreisachModel

for _mod in (
    SelfAdaptivePreisach,
    SelfAdaptivePreisachModel,
    DifferentiablePreisach,
    DifferentiablePreisachModel,
    DifferentiablePreisachNNModel,
    DifferentiablePreisachNN,
    EncoderDecoderPreisachNN,
    EncoderDecoderPreisachNNModel,
    BaseModule,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname


__all__ = [
    "BaseModule",
    "DifferentiablePreisach",
    "DifferentiablePreisachModel",
    "DifferentiablePreisachNN",
    "DifferentiablePreisachNNModel",
    "EncoderDecoderPreisachNN",
    "EncoderDecoderPreisachNNModel",
    "SelfAdaptivePreisach",
    "SelfAdaptivePreisachModel",
]
