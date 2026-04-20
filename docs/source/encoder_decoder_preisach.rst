============================
EncoderDecoderPreisachNN
============================

This document describes the information flow, constraints, auxiliary losses, and
regularization terms in ``EncoderDecoderPreisachNN``.

Overview
--------

The model maps a historical sequence of (H, B) measurements (the encoder context)
and a future sequence of H values (the decoder target) to predicted B values, using
a differentiable Preisach hysteresis operator as the core inductive bias.

All quantities are MinMax-normalized to [0, 1]:

- ``H_norm = 0`` corresponds to ``-H_max`` (negative saturation field)
- ``H_norm = 0.5`` corresponds to ``H = 0``
- ``H_norm = 1`` corresponds to ``+H_max``
- Same convention for ``B_norm``

The Preisach plane is defined on the unit triangle ``{(β, α) : 0 ≤ β ≤ α ≤ 1}``,
where ``α`` is the hysteron switch-up threshold and ``β`` is the switch-down threshold.

Information flow
----------------

::

    encoder_input  [B, T_enc, 2]          decoder_input  [B, T_dec, 1]
    (H_norm, B_norm history)               (future H_norm)
           |                                      |
           |                                      |
           v                                      |
    ┌─────────────────┐   mesh_coords             |
    │  PreisachEncoder │<──────────────────────┐  |
    │  (LSTM + cross-  │   [B, N, 3]           │  |
    │   attention)     │   (β, α, μ)           │  |
    └────────┬────────┘                        │  |
             │                                 │  |
             │  initial_states s₀              │  |
             │  [B, N]  ∈ [-1, 1]             │  |
             │                                 │  |
             v                                 │  |
    ┌─────────────────┐                        │  |
    │  get_states      │<── y0  (last H_norm   │  |
    │  (Preisach sweep │    from encoder)      │  |
    │   operator)      │<── decoder H_norm ────┘  |
    └────────┬────────┘<── α, β from mesh         |
             │                                    |
             │  states  [B, T_dec, N]             |
             │  s(α,β,t) ∈ [-1, 1]               |
             v                                    |
    ┌─────────────────┐                           |
    │  ResNetMLP       │<── mesh_coords [B, N, 2] |
    │  (density net)   │                          |
    └────────┬────────┘                           |
             │                                    |
             │  density μ(α,β)  [B, N]            |
             │                                    |
             v                                    |
    M = Σ μ·s / Σ μ   [B, T_dec]                 |
             |                                    |
             v                                    |
    B_norm = m_scale · M + m_offset   [B, T_dec] |
             |                                    |
             v                                    |
          y_hat  (predicted B_norm)               |

The mesh coordinates ``(β, α)`` are shared across the encoder, density network, and
Preisach sweep operator. During training, a small Gaussian perturbation is added to
mesh coordinates to regularize the density network. Perturbation is disabled when
``initial_states`` are passed in externally (e.g. during multi-window rollout) to keep
state indices aligned with the base mesh.

Three-phase training schedule
------------------------------

Training is split into three phases controlled by ``encoder_fit_steps`` and
``density_fit_steps``.

**Phase 1** (steps 0 → ``encoder_fit_steps``):

Only the encoder is trained. The density network is not yet reliable, so a fixed
``mock_density`` (see below) is used everywhere density appears — both in the M
computation inside the forward pass and in the aux loss. This prevents the random
density network from corrupting encoder gradients.

**Phase 2** (steps ``encoder_fit_steps`` → ``encoder_fit_steps + density_fit_steps``):

Both encoder and density network are trained jointly. The learned density replaces
``mock_density`` everywhere.

**Phase 3** (steps beyond phase 2 end):

Encoder and density network are trained in alternating steps (even step → encoder,
odd step → density). This prevents the two networks from chasing each other and
improves convergence stability.

Mock density
------------

During phase 1 the density network is randomly initialized and produces meaningless
output. Using it to weight the mean-field aux loss would give noisy, misleading
gradients to the encoder.

Instead, a fixed ``mock_density`` is precomputed at model initialization and registered
as a buffer (so it moves to the correct device automatically)::

    mock_density[i] = exp(-(α_i - β_i) / 0.1)
    mock_density    = mock_density / sum(mock_density)

This places most weight near the diagonal (``α ≈ β``), i.e. on easy/soft hysterons
that switch at low field differences. This is a physically reasonable prior for soft
magnetic materials like ARMCO iron, where most of the domain population is clustered
near zero coercivity.

Constraints and losses
-----------------------

Physics constraint (phase 1 only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a known last encoder field ``H_last``, two regions of the Preisach plane are
unambiguous regardless of magnetic history:

- ``β < H_last``: deactivation threshold is below the current field → hysteron is ON (+1)
- ``α > H_last``: activation threshold is above the current field → hysteron is OFF (-1)

The encoder is trained with an MSE loss against these deterministic targets, using
stratified averaging across the ON and OFF regions so that the gradient is balanced
regardless of how many hysterons fall in each region::

    loss_physics = 0.5 * (MSE(s₀[β<H], +1) + MSE(s₀[α>H], -1))

Auxiliary mean-field loss (all phases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The density-weighted mean of the initial states should match the observed B at the
end of the encoder context, inverted through the constitutive relation::

    M_target = (B_last - m_offset) / m_scale
    M_encoder = Σ μ·s₀ / Σ μ
    loss_aux = MSE(M_encoder, M_target)

In phase 1, ``μ = mock_density``. In phase 2+, ``μ`` is the learned density.

Saturation regularizer (all phases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without regularization the encoder can collapse to all ``s₀ = +1`` or all ``s₀ = -1``,
which satisfies many aggregated losses trivially (degenerate local minimum). A
penalty on ``mean(s₀²)`` pushes states toward the interior of [-1, 1]::

    loss_sat = mean(s₀²)

This does not prevent the states from reaching ±1 when the physics requires it — it
just removes the degenerate flat basin.

Symmetry regularizer (optional, off by default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   This regularizer assumes the material has a **symmetric major hysteresis loop**
   (equal positive and negative saturation, same loop shape on both branches).
   Enabling it for asymmetric materials will corrupt the learned density.
   Set ``symmetry_reg_weight > 0`` only after verifying your data satisfies this
   assumption.

For a symmetric material, a hysteron that switches up at ``α`` and down at ``β`` must
have a mirror that switches up at ``1 - β`` and down at ``1 - α`` (its negative-field
counterpart). This implies::

    μ(β, α) = μ(1 - α, 1 - β)

The regularizer penalizes the MSE between the density at each mesh point and the
density evaluated at its mirror::

    loss_sym = MSE(μ(β, α),  μ(1 - α, 1 - β).detach())

Gradients flow only through the left-hand side (the ``.detach()`` on the mirror
prevents circular gradients). Enable with ``symmetry_reg_weight > 0``; a
``UserWarning`` is raised at construction time as a reminder of the assumption.

Combined loss
~~~~~~~~~~~~~

::

    Phase 1:
        loss = loss_physics
             + aux_loss_weight    * loss_aux
             + saturation_reg_weight * loss_sat
             + symmetry_reg_weight   * loss_sym   # 0 by default

    Phase 2+:
        loss = MSE(y_hat, B_target)
             + aux_loss_weight    * loss_aux
             + saturation_reg_weight * loss_sat
             + symmetry_reg_weight   * loss_sym   # 0 by default

Multi-window rollout
---------------------

During validation the callback ``PlotHysteresisCallback`` produces a continuous
rollout by stitching consecutive dataset windows. At the boundary between window
``i`` and window ``i+1``, the terminal hysteron states from window ``i`` are passed
as ``initial_states`` to window ``i+1``, bypassing the encoder. This eliminates the
discontinuities that arise when each window reinitializes from its own encoder output.

The terminal states are computed externally using ``get_states`` on the decoder H
sequence, so the model forward signature does not need to return the full states
tensor.
