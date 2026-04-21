from __future__ import annotations

import torch

from ._states import sweep_left, sweep_up

__all__ = ["get_batched_states", "initialize_batched_state"]

CPU = torch.device("cpu")


def initialize_batched_state(
    batch_size: int,
    n_mesh_points: int,
    device: torch.device = CPU,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize batched states at negative saturation.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    n_mesh_points : int
        Number of mesh points in the Preisach model.
    device : torch.device
        Device to create the state tensor on. Must match the device of the input.
    dtype : torch.dtype
        Data type of the state tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Batched initial states and fields of shape [batch_size, n_mesh_points] and [batch_size, 1]
    """
    state = -torch.ones(batch_size, n_mesh_points, dtype=dtype, device=device)
    field = torch.zeros(batch_size, 1, dtype=dtype, device=device)
    return state, field


def get_batched_states(
    h: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_states: torch.Tensor | None = None,
    initial_fields: torch.Tensor | None = None,
    *,
    temp: float = 1e-3,
    dtype: torch.dtype = torch.float32,
    training: bool = True,
) -> torch.Tensor:
    """
    Compute batched active states of the Preisach model given applied fields and mesh points.

    This is a batched version of the get_states function that can handle multiple
    sequences simultaneously with different initial conditions per batch element.

    Parameters
    ----------
    h : torch.Tensor
        Applied field sequences of shape [batch_size, seq_len].
    alpha : torch.Tensor
        Hysteron activation points, shape [batch_size, n_mesh_points].
    beta : torch.Tensor
        Hysteron deactivation points, shape [batch_size, n_mesh_points].
    initial_states : torch.Tensor | None
        Initial state of the hysterons, shape [batch_size, n_mesh_points].
        If None, a default negative saturation state is used.
    initial_fields : torch.Tensor | None
        Initial field of the hysterons, shape [batch_size, 1].
        If None, a default zero field is used.
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
        States of each hysteron at each applied field, shape [batch_size, seq_len, n_mesh_points]
    """
    # Verify inputs are in normalized domain
    epsilon = 1e-6
    if torch.any(h + epsilon < 0) or torch.any(h - epsilon > 1):
        msg = "Applied field values are outside of the unit domain"
        raise RuntimeError(msg)

    batch_size, seq_len = h.shape
    n_mesh_points = alpha.shape[1]

    # Initialize states if not provided
    if initial_states is None or initial_fields is None:
        if initial_states is not None or initial_fields is not None:
            msg = "Both initial_states and initial_fields must be provided or both must be None"
            raise ValueError(msg)
        initial_states, initial_fields = initialize_batched_state(
            batch_size, n_mesh_points, device=h.device, dtype=dtype
        )

    # Validate input shapes
    if initial_states.shape != (batch_size, n_mesh_points):
        msg = f"initial_states must have shape [{batch_size}, {n_mesh_points}]"
        raise ValueError(msg)
    if initial_fields.shape != (batch_size, 1):
        msg = f"initial_fields must have shape [{batch_size}, 1]"
        raise ValueError(msg)
    if alpha.shape != (batch_size, n_mesh_points):
        msg = f"alpha must have shape [{batch_size}, {n_mesh_points}]"
        raise ValueError(msg)
    if beta.shape != (batch_size, n_mesh_points):
        msg = f"beta must have shape [{batch_size}, {n_mesh_points}]"
        raise ValueError(msg)

    # Initialize output tensor for all states
    all_states = torch.zeros(
        batch_size, seq_len, n_mesh_points, dtype=dtype, device=h.device
    )

    # Track current state for each batch element
    current_states = initial_states.clone()  # [batch_size, n_mesh_points]
    previous_fields = initial_fields.squeeze(-1)  # [batch_size]

    # Process each time step
    for t in range(seq_len):
        current_fields = h[:, t]  # [batch_size]

        # Determine sweep direction for each batch element
        field_increase = current_fields > previous_fields  # [batch_size]
        field_decrease = current_fields < previous_fields  # [batch_size]
        ~(field_increase | field_decrease)  # [batch_size]

        # Apply sweep_up where field increased
        if field_increase.any():
            # Expand dimensions for broadcasting: [batch_size, 1] and [batch_size, n_mesh_points]
            h_expanded = current_fields[field_increase].unsqueeze(-1)  # [n_increase, 1]
            alpha_increase = alpha[field_increase]  # [n_increase, n_mesh_points]
            states_increase = current_states[
                field_increase
            ]  # [n_increase, n_mesh_points]

            new_states_increase = sweep_up(
                h_expanded,
                alpha_increase,
                states_increase,
                temp=temp,
                training=training,
            )
            current_states[field_increase] = new_states_increase

        # Apply sweep_left where field decreased
        if field_decrease.any():
            h_expanded = current_fields[field_decrease].unsqueeze(-1)  # [n_decrease, 1]
            beta_decrease = beta[field_decrease]  # [n_decrease, n_mesh_points]
            states_decrease = current_states[
                field_decrease
            ]  # [n_decrease, n_mesh_points]

            new_states_decrease = sweep_left(
                h_expanded, beta_decrease, states_decrease, temp=temp, training=training
            )
            current_states[field_decrease] = new_states_decrease

        # States remain the same where field didn't change (field_same case is automatic)

        # Store current states
        all_states[:, t, :] = current_states

        # Update previous fields
        previous_fields = current_fields

    return all_states
