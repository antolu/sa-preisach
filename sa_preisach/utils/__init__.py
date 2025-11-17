from .._mod_replace import replace_modname
from ._batched_states import get_batched_states, initialize_batched_state
from ._grad import set_requires_grad
from ._mesh import (
    DefaultMeshSizeFunction,
    constant_mesh_size,
    create_triangle_mesh,
    default_mesh_size,
    exponential_mesh,
    make_mesh_size_function,
)
from ._states import get_states, initialize_state, sweep_left, sweep_up, switch

for _mod in (
    default_mesh_size,
    DefaultMeshSizeFunction,
    constant_mesh_size,
    exponential_mesh,
    create_triangle_mesh,
    switch,
    initialize_state,
    get_states,
    get_batched_states,
    initialize_batched_state,
    sweep_up,
    sweep_left,
    make_mesh_size_function,
    set_requires_grad,
):
    replace_modname(_mod, __name__)


del _mod
del replace_modname


__all__ = [
    "DefaultMeshSizeFunction",
    "constant_mesh_size",
    "create_triangle_mesh",
    "default_mesh_size",
    "exponential_mesh",
    "get_batched_states",
    "get_states",
    "initialize_batched_state",
    "initialize_state",
    "make_mesh_size_function",
    "set_requires_grad",
    "sweep_left",
    "sweep_up",
    "switch",
]
