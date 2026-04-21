---
title: Self-adaptive loss weighting (SA-PINN) for EncoderDecoderPreisachNN
date: 2026-04-21
status: approved
---

## Context

The current model uses static scalar hyperparameters (`aux_loss_weight`, `saturation_reg_weight`, and per-prior `weight` floats) to combine loss terms. Tuning these by hand is tedious and fragile. The SA-PINN paper (arxiv 2009.04544) introduces learned log-weight parameters that are stepped in the **opposite direction** from the model — effectively doing automatic loss balancing by pushing weight mass toward whichever term the model is currently failing at.

## Scope

- Add SA-PINN adaptive loss weighting to `EncoderDecoderPreisachNN` with an on/off toggle
- Each loss term (seq, aux, sat, and each named prior term) gets its own learnable weight
- Static weights remain default; SA-PINN is opt-in via `adaptive_loss_weights=True`
- Log unweighted combined loss and individual unweighted terms in both modes
- Document in `docs/source/encoder_decoder_preisach.rst`

## Data structures

### `DensityPrior` base class changes

- Add `log_weight: nn.Parameter` initialized to `torch.log(torch.tensor(self.weight))`
- `forward()` returns **unweighted** loss values per key — the static `self.weight` multiplication moves to the call site in `common_step`
- This is required in both modes: SA-PINN needs raw terms to apply learned weights; it also enables logging unweighted individual losses

### `AdaptiveLossWeights(nn.Module)`

New small module in `sa_preisach/models/_adaptive_loss_weights.py`:

```python
class AdaptiveLossWeights(nn.Module):
    def __init__(self, aux_loss_weight: float, saturation_reg_weight: float) -> None:
        super().__init__()
        self.log_seq = nn.Parameter(torch.tensor(0.0))
        self.log_aux = nn.Parameter(torch.log(torch.tensor(aux_loss_weight)))
        self.log_sat = nn.Parameter(torch.log(torch.tensor(saturation_reg_weight)))
```

Registered on the Lightning module as `self.adaptive_weights` when `adaptive_loss_weights=True`, else `None`. Checkpointed under `adaptive_weights.*`.

### Prior `log_weight` parameters

Each `DensityPrior` leaf owns its `log_weight: nn.Parameter`. These are checkpointed under the prior's existing submodule path (e.g. `model.density_prior.priors.0.log_weight`). The SA-PINN optimizer collects them via an extended `_inject_density_net` that also populates `self._prior_leaves: list[DensityPrior]` — a flat list of non-composite prior nodes, reused by both `configure_optimizers` and `common_step`. Only leaf priors get a `log_weight`; `CompositeDensityPrior` nodes do not.

## Loss computation in `common_step`

**SA-PINN off:**
- `prior_losses` values multiplied by `prior_leaf.weight` (static float) at call site
- `aux_loss_weight` and `saturation_reg_weight` applied as today
- No behaviour change

**SA-PINN on:**
- All individual loss terms computed unweighted
- Weighted combination: `exp(log_w) * loss` per term
- Prior terms: `exp(prior_leaf.log_weight) * unweighted_prior_term`
- `loss_unweighted` = simple sum of all unweighted terms, logged but not used for backward
- `loss_weighted` = SA-PINN weighted sum, used for `manual_backward`

Both paths return the same dict keys so callbacks are unaffected.

## Optimizer and stepping

Fourth optimizer added in `configure_optimizers` when `adaptive_loss_weights=True`:

```python
optimizer_adaptive = AdamW(
    list(self.adaptive_weights.parameters())
    + [leaf.log_weight for leaf in prior_leaves],
    lr=lr_adaptive,
)
scheduler_adaptive = StepLR(optimizer_adaptive, step_size=lr_step_interval, gamma=lr_gamma)
```

**Opposite-direction stepping:** after `manual_backward(weighted_loss)`, negate gradients before stepping:

```python
for p in adaptive_params:
    if p.grad is not None:
        p.grad.neg_()
optimizer_adaptive.step()
```

**Phase gating** via `adaptive_loss_start: Literal["all_phases", "phase2_plus"]`:
- `"all_phases"`: `optimizer_adaptive` steps whenever any model optimizer steps
- `"phase2_plus"`: `optimizer_adaptive` steps only when `step >= phase1_end`

The `optimizer_step` function receives `optimizer_adaptive` and `adaptive_loss_start` as arguments and resolves the gating logic internally.

## New hyperparameters

```python
adaptive_loss_weights: bool = False
adaptive_loss_start: Literal["all_phases", "phase2_plus"] = "phase2_plus"
lr_adaptive: float = 1e-3
```

`adaptive_loss_weights=False` is a strict no-op: no new parameters, no new optimizers, identical checkpoint format.

## Logging

When `adaptive_loss_weights=True`:

| Metric | Where |
|---|---|
| `train/loss_unweighted` | sum of all unweighted terms |
| `train/adaptive_weight/seq` | `exp(log_seq)` |
| `train/adaptive_weight/aux` | `exp(log_aux)` |
| `train/adaptive_weight/sat` | `exp(log_sat)` |
| `train/adaptive_weight/prior/<key>` | `exp(prior_leaf.log_weight)` per leaf |
| `validation/loss_unweighted` | same, on validation |

Existing per-term logs (`train/aux_loss`, `train/physics_loss`, `train/mse`, etc.) are unchanged and represent unweighted values in both modes.

## Documentation

New section **"Self-adaptive loss weighting"** added to `docs/source/encoder_decoder_preisach.rst` after "Combined loss", covering:
- The SA-PINN mechanism and opposite-direction stepping
- The three new hparams and defaults
- Phase gating options and guidance on when to use each
- Updated combined loss block showing both static and adaptive formulations
- Note that `DensityPrior.forward()` returns unweighted losses in both modes

## Files to create/modify

| File | Change |
|---|---|
| `sa_preisach/models/_adaptive_loss_weights.py` | New: `AdaptiveLossWeights` module |
| `sa_preisach/priors/_base.py` | Add `log_weight: nn.Parameter` |
| `sa_preisach/priors/_symmetry.py` | Remove static weight multiplication from `forward()` |
| `sa_preisach/models/_encoder_decoder_preisach_nn.py` | SA-PINN integration throughout |
| `docs/source/encoder_decoder_preisach.rst` | New section + updated combined loss block |
