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

The regularizer penalizes the density-weighted squared difference between each mesh
point and its mirror. Plain MSE is diluted when density is concentrated on few points
(near-zero regions contribute nothing meaningful). Weighting by the average density
ensures the penalty fires where mass actually exists::

    w = (μ(β, α) + μ(1-α, 1-β)) / 2
    loss_sym = Σ w · (μ(β, α) - μ(1-α, 1-β))² / Σ w

Gradients flow only through the left-hand ``μ(β, α)`` (the mirror is detached).
Enable with ``symmetry_reg_weight > 0``; a ``UserWarning`` is raised at construction
time as a reminder of the assumption.

Combined loss
~~~~~~~~~~~~~

**Static mode** (``adaptive_loss_weights=False``, default)::

    Phase 1:
        loss = loss_physics
             + aux_loss_weight        * loss_aux
             + saturation_reg_weight  * loss_sat
             + Σ prior_leaf.weight    * loss_prior_i

    Phase 2+:
        loss = MSE(y_hat, B_target)
             + aux_loss_weight        * loss_aux
             + saturation_reg_weight  * loss_sat
             + Σ prior_leaf.weight    * loss_prior_i

**Adaptive mode** (``adaptive_loss_weights=True``)::

    Phase 1:
        loss = exp(λ_seq) * loss_physics
             + exp(λ_aux) * loss_aux
             + exp(λ_sat) * loss_sat
             + Σ exp(λ_prior_i) * loss_prior_i

    Phase 2+:
        loss = exp(λ_seq) * MSE(y_hat, B_target)
             + exp(λ_aux) * loss_aux
             + exp(λ_sat) * loss_sat
             + Σ exp(λ_prior_i) * loss_prior_i

    where λ_* are learned log-scale noise parameters (see Self-adaptive loss weighting).

Self-adaptive loss weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enabled with ``adaptive_loss_weights=True``. Each loss term is assigned a learned
log-scale noise parameter ``log_s`` (an ``nn.Parameter``), and the combined loss uses
the Kendall & Gal homoscedastic uncertainty formulation:

.. math::

    \mathcal{L} = \sum_i \left( \frac{L_i}{2 \exp(2 \log s_i)} + \log s_i \right)

The ``log_s`` penalty term is self-regularizing: increasing ``log_s`` down-weights the
scaled loss but pays an additive ``log_s`` cost, creating a natural equilibrium. No
gradient negation or weight decay is needed.

Hyperparameters
^^^^^^^^^^^^^^^

``adaptive_loss_weights`` (bool, default ``False``)
    Enable or disable the mechanism. When ``False``, all existing static weight
    behaviour is unchanged and no new parameters are added to the checkpoint.

``adaptive_loss_start`` (``"all_phases"`` | ``"phase2_plus"``, default ``"phase2_plus"``)
    When to activate the adaptive optimizer.

    - ``"phase2_plus"``: adaptive weights are updated only from phase 2 onwards
      (once the density network starts training). Recommended for most runs — in
      phase 1 only the encoder is trained and having the weights adapt to the
      encoder-only loss landscape can bias them before density training begins.
    - ``"all_phases"``: adaptive weights update from the very first step. Use
      when you want the weights to adapt during phase 1 as well, e.g. to balance
      ``physics_loss`` and ``aux_loss`` automatically.

``lr_adaptive`` (float, default ``1e-3``)
    Learning rate for the adaptive weight optimizer (``AdamW``).

Unweighted logging
^^^^^^^^^^^^^^^^^^

When SA-PINN is active, the following additional metrics are logged:

- ``train/loss_unweighted``, ``validation/loss_unweighted``: simple sum of all
  unweighted loss terms (no learned weights). Use this to verify that the raw
  combined loss is decreasing even as the weights shift.
- ``train/adaptive_weight/seq``, ``train/adaptive_weight/aux``,
  ``train/adaptive_weight/sat``: effective weight (``exp(λ)`` in natural scale)
  for the sequence, auxiliary, and saturation terms.
- ``train/adaptive_weight/prior/<key>``: effective weight per named prior term.

Individual unweighted losses (``train/aux_loss``, ``train/physics_loss``,
``train/mse``, etc.) are always logged in natural scale regardless of mode.

Note on ``DensityPrior``
^^^^^^^^^^^^^^^^^^^^^^^^^

``DensityPrior.forward()`` always returns **unweighted** loss values. The static
``weight`` float on each prior (used when ``adaptive_loss_weights=False``) is
applied at the ``common_step`` call site, not inside ``forward()``. The learned
``log_weight`` parameter on each leaf prior is used when SA-PINN is active.

Density priors
--------------

``DensityPrior`` subclasses add soft constraints on the learned density distribution
``μ(β, α)``. They are composed with ``CompositeDensityPrior`` and passed via the
``density_prior`` argument. Each prior returns **unweighted** loss values; weighting
is applied at the ``common_step`` call site (static ``weight`` float, or learned
``log_weight`` in SA-PINN mode).

All priors inherit ``log_weight: nn.Parameter`` from ``DensityPrior`` and are
automatically included in the adaptive optimizer when ``adaptive_loss_weights=True``.

``DiagonalDensityPrior``
~~~~~~~~~~~~~~~~~~~~~~~~

Penalizes density mass far from the ``α = β`` diagonal::

    loss = mean( Σ μᵢ (αᵢ - βᵢ)² / Σ μᵢ )

Pushes hysterons toward small coercivity (narrow loops). Universally useful as a
baseline regularizer.

``CentroidDensityPrior``
~~~~~~~~~~~~~~~~~~~~~~~~

Penalizes the density-weighted centroid being far from the origin::

    loss = mean( Σ μᵢ (αᵢ + βᵢ)/2 / Σ μᵢ )

Pulls mass toward small ``(α, β)`` — hysterons that flip at low fields. Use for
soft magnetic materials where most hysteretic activity occurs near zero field.

``BoundaryDensityPrior``
~~~~~~~~~~~~~~~~~~~~~~~~

Penalizes density mass near the triangle boundary via an exponential proximity term::

    margin_i = min(α_i,  1 - β_i,  α_i - β_i)
    loss = mean( Σ μᵢ exp(-margin_i / σ) / Σ μᵢ )

The ``sigma`` parameter (default ``0.05``) controls how close to the boundary the
penalty activates. Prevents degenerate solutions where hysterons cluster at the
corners or edges of the Preisach triangle.

``EntropyDensityPrior``
~~~~~~~~~~~~~~~~~~~~~~~

Maximizes the entropy of the normalized density distribution::

    p_i = μᵢ / Σ μⱼ
    loss = -mean( Σ p_i log(p_i) )     [negated so minimizing reduces entropy loss]

Encourages a spread-out coercivity distribution. Useful as a regularizer against
delta-like collapse to a single hysteron cluster.

Material-specific combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------------------------------------+
| Material type             | Suggested priors                         |
+===========================+==========================================+
| Soft magnet, low-field    | ``Diagonal + Centroid``                  |
+---------------------------+------------------------------------------+
| Soft magnet, broad dist.  | ``Diagonal + Entropy``                   |
+---------------------------+------------------------------------------+
| Any, avoid degeneracy     | ``+ Boundary``                           |
+---------------------------+------------------------------------------+
| Symmetric loop            | ``+ Symmetry``                           |
+---------------------------+------------------------------------------+

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
