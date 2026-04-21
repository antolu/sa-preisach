from __future__ import annotations

import torch

CPU = torch.device("cpu")


def switch(
    h: torch.Tensor, mesh: torch.Tensor, *, temp: float = 1e-4, training: bool = True
) -> torch.Tensor:
    """
    Differentiable hysteron switch. Outputs 2.0 for h > mesh, 0.0 for h < mesh, and
    1.0 for h == mesh. The switch is smooth and differentiable for training, but can be
    sharp for inference.

    Parameters
    ----------
    h : torch.Tensor
        Applied field.
    mesh : torch.Tensor
        Mesh points (hysteron activation and deactivation points).
    temp : float
        Temperature parameter for the switch. Higher values make the switch smoother.
    training : bool
        If False, a sharp switch is used. If True, a differentiable switch is used.
    """
    if training:
        # Differentiable tanh-based switch
        return 1.0 + torch.tanh((h - mesh - 0 * temp) / abs(temp))

    # Non-differentiable step function using torch.heaviside
    return 2.0 * torch.heaviside(
        h - mesh, torch.tensor(0.0, dtype=h.dtype, device=h.device)
    )


def sweep_up(
    h: torch.Tensor,
    alpha: torch.Tensor,
    prev_state: torch.Tensor,
    *,
    temp: float = 1e-2,
    training: bool = True,
) -> torch.Tensor:
    return torch.minimum(
        prev_state + switch(h, alpha, temp=temp, training=training),
        torch.ones_like(alpha, dtype=alpha.dtype, device=alpha.device) * 1.0,
    )


def sweep_left(
    h: torch.Tensor,
    beta: torch.Tensor,
    prev_state: torch.Tensor,
    *,
    temp: float = 1e-2,
    training: bool = True,
) -> torch.Tensor:
    return torch.maximum(
        prev_state - switch(beta, h, temp=temp, training=training),
        -torch.ones_like(beta, dtype=beta.dtype, device=beta.device),
    )


def initialize_state(
    n_mesh_points: int,
    device: torch.device = CPU,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize states at negative saturation.

    Parameters
    ----------
    n_mesh_points : int
        Number of mesh points in the Preisach model.
    device : torch.device
        Device to create the state tensor on. Must match the device of the input.
    dtype : torch.dtype
        Data type of the state tensor.
    """
    state = -torch.ones(n_mesh_points, dtype=dtype, device=device)
    field = torch.zeros(1, dtype=dtype, device=device)
    return state, field


def get_states(  # noqa: PLR0912
    h: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    current_state: torch.Tensor | None = None,
    current_field: torch.Tensor | float | None = None,
    *,
    temp: float = 1e-3,
    dtype: torch.dtype = torch.float32,
    training: bool = True,
) -> torch.Tensor:
    """
    Compute active states of the Preisach model given the applied field and mesh points.

    Parameters
    ----------
    h : torch.Tensor
        Applied field.
    alpha : torch.Tensor
        Hysteron activation points, shape (N)
    beta : torch.Tensor
        Hysteron deactivation points, shape (N)
    current_state : torch.Tensor | None
        Current state of the hysterons, shape (N). If None, a default negative saturation
        state is used.
    current_field : torch.Tensor | float | None
        Current field of the hysterons, shape (1). If None, a default zero field is used.
    temp : float
        Temperature parameter for the hysteron switch. Higher values make the switch
        smoother.
    dtype : torch.dtype
        Data type of the state tensor.
    training : bool
        If False, a sharp switch is used. If True, a differentiable switch is used.

    Returns
    -------
    torch.Tensor
        States of each hysteron at each applied field, shape (N, M), where M is the number of
        applied fields and N is the number of hysterons.
    """
    # verify the inputs are in the normalized region within some machine epsilon
    epsilon = 1e-6
    if torch.any(h + epsilon < 0) or torch.any(h - epsilon > 1):
        msg = "applied values are outside of the unit domain"
        raise RuntimeError(msg)

    assert len(h.shape) == 1
    n_mesh_points = alpha.shape[0]

    # list of hysteresis states with initial state set
    if current_state is None and current_field is None:
        initial_state, initial_field = initialize_state(
            n_mesh_points,
            device=h.device,
            dtype=dtype,
        )
    elif current_state is None and current_field is not None:
        msg = "current_state is None but current_field is not None"
        raise ValueError(msg)
    elif current_state is not None and current_field is None:
        msg = "current_field is None but current_state is not None"
        raise ValueError(msg)
    else:
        initial_state = current_state
        initial_field = current_field

    assert initial_state is not None
    assert initial_field is not None

    states = []

    # loop through the states
    for i in range(len(h)):
        if i == 0:
            # handle initial case
            if h[0] > initial_field:
                states += [
                    sweep_up(h[i], alpha, initial_state, temp=temp, training=training)
                ]
            elif h[0] < initial_field:
                states += [
                    sweep_left(h[i], beta, initial_state, temp=temp, training=training)
                ]
            else:
                states += [initial_state]

        elif h[i] > h[i - 1]:
            # if the new applied field is greater than the old one, sweep up to
            # new applied field
            states += [
                sweep_up(h[i], alpha, states[i - 1], temp=temp, training=training)
            ]
        elif h[i] < h[i - 1]:
            # if the new applied field is less than the old one, sweep left to
            # new applied field
            states += [
                sweep_left(h[i], beta, states[i - 1], temp=temp, training=training)
            ]
        else:
            states += [states[i - 1]]

    # concatenate states into one tensor
    return torch.cat([ele.unsqueeze(0) for ele in states])
