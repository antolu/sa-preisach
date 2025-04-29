from .._mod_replace import replace_modname
from ._mesh import (
    constant_mesh_size,
    create_triangle_mesh,
    default_mesh_size,
    exponential_mesh,
)
from ._states import get_states, initialize_state, sweep_left, sweep_up, switch

for _mod in (
    default_mesh_size,
    constant_mesh_size,
    exponential_mesh,
    create_triangle_mesh,
    switch,
    initialize_state,
    get_states,
    sweep_up,
    sweep_left,
):
    replace_modname(_mod, __name__)


del _mod
del replace_modname


__all__ = [
    "constant_mesh_size",
    "create_triangle_mesh",
    "default_mesh_size",
    "exponential_mesh",
    "get_states",
    "initialize_state",
    "sweep_left",
    "sweep_up",
    "switch",
]
