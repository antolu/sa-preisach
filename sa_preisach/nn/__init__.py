from .._mod_replace import replace_modname
from ._binary_parameter import BinaryParameter
from ._constrained_parameter import ConstrainedParameter
from ._gpy_constrained_parameter import GPyConstrainedParameter
from ._preisach_encoder import PreisachEncoder
from ._preisach_gru_encoder import PreisachGRUEncoder
from ._preisach_lstm_encoder import PreisachLSTMEncoder
from ._preisach_rnn_encoder import PreisachRNNEncoder
from ._preisach_transformer_encoder import PreisachTransformerEncoder
from ._resnet import ResNetMLP
from ._smooth_switch import SmoothSwitch

for _mod in (
    BinaryParameter,
    ConstrainedParameter,
    GPyConstrainedParameter,
    PreisachEncoder,
    PreisachGRUEncoder,
    PreisachLSTMEncoder,
    PreisachRNNEncoder,
    PreisachTransformerEncoder,
    ResNetMLP,
    SmoothSwitch,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "BinaryParameter",
    "ConstrainedParameter",
    "GPyConstrainedParameter",
    "PreisachEncoder",
    "PreisachGRUEncoder",
    "PreisachLSTMEncoder",
    "PreisachRNNEncoder",
    "PreisachTransformerEncoder",
    "ResNetMLP",
    "SmoothSwitch",
]
