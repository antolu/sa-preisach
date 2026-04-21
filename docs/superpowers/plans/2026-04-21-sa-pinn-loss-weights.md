# SA-PINN Adaptive Loss Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-adaptive loss weighting (SA-PINN style) to `EncoderDecoderPreisachNN` where each loss term has a learned log-weight parameter stepped in the opposite direction from the model, with an on/off toggle and full logging of unweighted losses and adaptive weights.

**Architecture:** Each `DensityPrior` leaf owns a `log_weight: nn.Parameter`; a new `AdaptiveLossWeights` module owns `log_seq/log_aux/log_sat`; a fourth optimizer steps all of these with negated gradients. Static weight behaviour is preserved when `adaptive_loss_weights=False`.

**Tech Stack:** PyTorch, PyTorch Lightning, existing `EncoderDecoderPreisachNN` / `DensityPrior` class hierarchy.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `sa_preisach/priors/_base.py` | Modify | Add `log_weight: nn.Parameter`; `forward()` returns unweighted losses |
| `sa_preisach/priors/_diagonal.py` | Modify | Remove `self.weight *` from return value |
| `sa_preisach/priors/_symmetry.py` | Modify | Remove `self.weight *` from return value |
| `sa_preisach/priors/_composite.py` | No change | Already just aggregates child outputs |
| `sa_preisach/models/_adaptive_loss_weights.py` | Create | `AdaptiveLossWeights(nn.Module)` with `log_seq/log_aux/log_sat` |
| `sa_preisach/models/__init__.py` | Modify | Export `AdaptiveLossWeights` |
| `sa_preisach/models/_encoder_decoder_preisach_nn.py` | Modify | Wire everything: hparams, optimizers, `common_step`, logging |
| `docs/source/encoder_decoder_preisach.rst` | Modify | New section + updated combined loss block |
| `tests/test_encoder_decoder_preisach_nn.py` | Modify | Tests for adaptive mode on/off, weight negation, logging keys |
| `tests/test_density_priors.py` | Create | Tests for unweighted `forward()`, `log_weight` param existence |

---

### Task 1: Make `DensityPrior.forward()` return unweighted losses

**Files:**
- Modify: `sa_preisach/priors/_base.py`
- Modify: `sa_preisach/priors/_diagonal.py`
- Modify: `sa_preisach/priors/_symmetry.py`
- Create: `tests/test_density_priors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_density_priors.py
from __future__ import annotations

import torch
import pytest
from sa_preisach.priors import DiagonalDensityPrior, SymmetryDensityPrior, DensityPrior


def _mesh_and_density(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = torch.tensor([[0.0, 0.2], [0.1, 0.5], [0.3, 0.7], [0.5, 1.0]])
    density = torch.ones(1, n)
    mesh = mesh.unsqueeze(0)  # [1, 4, 2]
    return mesh, density


def test_diagonal_prior_has_log_weight_parameter() -> None:
    prior = DiagonalDensityPrior(weight=2.0)
    assert hasattr(prior, "log_weight")
    assert isinstance(prior.log_weight, torch.nn.Parameter)
    assert torch.isclose(prior.log_weight.exp(), torch.tensor(2.0), atol=1e-5)


def test_symmetry_prior_has_log_weight_parameter() -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior = SymmetryDensityPrior(weight=3.0)
    assert hasattr(prior, "log_weight")
    assert isinstance(prior.log_weight, torch.nn.Parameter)
    assert torch.isclose(prior.log_weight.exp(), torch.tensor(3.0), atol=1e-5)


def test_diagonal_forward_returns_unweighted_loss() -> None:
    prior = DiagonalDensityPrior(weight=5.0)
    mesh, density = _mesh_and_density()
    out = prior(mesh, density)
    assert "diagonal" in out
    # With weight=5 the old code would return 5*loss; now it should return just loss
    # Verify by checking that the value does NOT change when weight changes
    prior2 = DiagonalDensityPrior(weight=10.0)
    out2 = prior2(mesh, density)
    assert torch.isclose(out["diagonal"], out2["diagonal"], atol=1e-6)


def test_symmetry_forward_returns_unweighted_loss() -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior = SymmetryDensityPrior(weight=5.0)
        prior2 = SymmetryDensityPrior(weight=10.0)

    def mock_density_net(coords: torch.Tensor) -> torch.Tensor:
        return torch.ones(coords.shape[0], coords.shape[1]) * 0.5

    prior.density_net = mock_density_net
    prior2.density_net = mock_density_net
    mesh, density = _mesh_and_density()

    out = prior(mesh, density)
    out2 = prior2(mesh, density)
    assert torch.isclose(out["symmetry"], out2["symmetry"], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_density_priors.py -v
```

Expected: FAIL — `log_weight` attribute missing, and weight still multiplied in forward.

- [ ] **Step 3: Add `log_weight` to `DensityPrior` base and update docstring**

Replace `sa_preisach/priors/_base.py` with:

```python
from __future__ import annotations

import abc
import math

import torch


class DensityPrior(torch.nn.Module, abc.ABC):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.log_weight = torch.nn.Parameter(torch.tensor(math.log(weight)))

    @abc.abstractmethod
    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mesh_coords : torch.Tensor
            Mesh coordinates [batch, N, 2] where [..., 0] is beta and [..., 1] is alpha.
        density : torch.Tensor
            Density values [batch, N].

        Returns
        -------
        dict[str, torch.Tensor]
            Named scalar loss terms, **unweighted**. The caller is responsible for
            applying either the static ``self.weight`` float or the learned
            ``self.log_weight`` parameter depending on the training mode.
        """
```

- [ ] **Step 4: Remove `self.weight *` from `DiagonalDensityPrior.forward()`**

In `sa_preisach/priors/_diagonal.py`, change the return line:

```python
        return {"diagonal": loss}
```

- [ ] **Step 5: Remove `self.weight *` from `SymmetryDensityPrior.forward()`**

In `sa_preisach/priors/_symmetry.py`, change the return line:

```python
        return {"symmetry": loss}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_density_priors.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add sa_preisach/priors/_base.py sa_preisach/priors/_diagonal.py sa_preisach/priors/_symmetry.py tests/test_density_priors.py
git commit -m "refactor(priors): return unweighted losses from forward(), add log_weight parameter"
```

---

### Task 2: Update `common_step` call site to apply static weights to prior losses

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

The `prior_losses` dict now contains unweighted values. The existing call site that sums them must apply `self.weight` explicitly, so static-mode behaviour is preserved.

- [ ] **Step 1: Write a failing test for static-mode prior weight application**

Add to `tests/test_density_priors.py`:

```python
def test_composite_prior_forward_aggregates_unweighted() -> None:
    from sa_preisach.priors import CompositeDensityPrior
    p1 = DiagonalDensityPrior(weight=2.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2 = SymmetryDensityPrior(weight=3.0)

    def mock_density_net(coords: torch.Tensor) -> torch.Tensor:
        return torch.ones(coords.shape[0], coords.shape[1]) * 0.5

    p2.density_net = mock_density_net
    composite = CompositeDensityPrior(p1, p2)
    mesh, density = _mesh_and_density()
    out = composite(mesh, density)
    # Both keys should be present and unweighted
    assert "diagonal" in out
    assert "symmetry" in out
    # Diagonal loss should equal DiagonalDensityPrior(weight=1.0) output
    p1_unit = DiagonalDensityPrior(weight=1.0)
    out_unit = p1_unit(mesh, density)
    assert torch.isclose(out["diagonal"], out_unit["diagonal"], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_density_priors.py::test_composite_prior_forward_aggregates_unweighted -v
```

Expected: FAIL (composite currently just passes through child outputs, which are now unweighted — this test should actually pass. Run to confirm).

- [ ] **Step 3: Fix `common_step` to apply static `prior_leaf.weight` at call site**

In `sa_preisach/models/_encoder_decoder_preisach_nn.py`, find the block starting at line ~702:

```python
        prior_losses: dict[str, torch.Tensor] = (
            self.model.density_prior(mesh_coords, density)
            if self.model.density_prior is not None
            else {}
        )
        prior_loss = (
            sum(prior_losses.values())  # type: ignore[arg-type]
            if prior_losses
            else torch.zeros(1, device=density.device)
        )
```

Replace with:

```python
        prior_losses_raw: dict[str, torch.Tensor] = (
            self.model.density_prior(mesh_coords, density)
            if self.model.density_prior is not None
            else {}
        )
        # Apply static weights at the call site (forward() returns unweighted losses).
        # _prior_leaves maps key → leaf prior so we can look up self.weight per term.
        prior_losses: dict[str, torch.Tensor] = {
            k: v * self._prior_leaf_by_key[k].weight
            for k, v in prior_losses_raw.items()
        } if prior_losses_raw else {}
        prior_loss = (
            sum(prior_losses.values())  # type: ignore[arg-type]
            if prior_losses
            else torch.zeros(1, device=density.device)
        )
```

Note: `self._prior_leaf_by_key` will be populated in Task 4. For now, add a placeholder that makes existing tests pass — we'll fully wire it in Task 4. Add this temporary line to `__init__` after the `density_prior` assignment:

```python
        self._prior_leaf_by_key: dict[str, DensityPrior] = {}
```

And add the import at the top of the file if not already present:
```python
from ..priors import DensityPrior
```

- [ ] **Step 4: Run existing tests to verify no regression**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py tests/test_density_priors.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_density_priors.py
git commit -m "refactor(model): apply static prior weights at call site in common_step"
```

---

### Task 3: Create `AdaptiveLossWeights` module

**Files:**
- Create: `sa_preisach/models/_adaptive_loss_weights.py`
- Modify: `sa_preisach/models/__init__.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def test_adaptive_loss_weights_initial_values() -> None:
    from sa_preisach.models._adaptive_loss_weights import AdaptiveLossWeights
    alw = AdaptiveLossWeights(aux_loss_weight=2.0, saturation_reg_weight=0.5)
    assert torch.isclose(alw.log_seq.exp(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(alw.log_aux.exp(), torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(alw.log_sat.exp(), torch.tensor(0.5), atol=1e-5)


def test_adaptive_loss_weights_parameters_are_nn_parameters() -> None:
    from sa_preisach.models._adaptive_loss_weights import AdaptiveLossWeights
    alw = AdaptiveLossWeights(aux_loss_weight=1.0, saturation_reg_weight=1.0)
    params = list(alw.parameters())
    assert len(params) == 3
    for p in params:
        assert isinstance(p, torch.nn.Parameter)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_initial_values tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_parameters_are_nn_parameters -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `AdaptiveLossWeights`**

```python
# sa_preisach/models/_adaptive_loss_weights.py
from __future__ import annotations

import math

import torch


class AdaptiveLossWeights(torch.nn.Module):
    def __init__(self, aux_loss_weight: float, saturation_reg_weight: float) -> None:
        super().__init__()
        self.log_seq = torch.nn.Parameter(torch.tensor(0.0))
        self.log_aux = torch.nn.Parameter(torch.tensor(math.log(aux_loss_weight)))
        self.log_sat = torch.nn.Parameter(torch.tensor(math.log(saturation_reg_weight)))
```

- [ ] **Step 4: Export from `sa_preisach/models/__init__.py`**

Add to imports and `__all__`:

```python
from ._adaptive_loss_weights import AdaptiveLossWeights
```

And add `"AdaptiveLossWeights"` to the `__all__` list. Also add it to the `for _mod in (...)` loop:

```python
for _mod in (
    AdaptiveLossWeights,
    SelfAdaptivePreisach,
    ...
):
    replace_modname(_mod, __name__)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_initial_values tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_parameters_are_nn_parameters -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add sa_preisach/models/_adaptive_loss_weights.py sa_preisach/models/__init__.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "feat(models): add AdaptiveLossWeights module"
```

---

### Task 4: Extend `_inject_density_net` to populate `_prior_leaf_by_key` and `_prior_leaves`

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

`_prior_leaf_by_key: dict[str, DensityPrior]` maps each loss key (e.g. `"diagonal"`, `"symmetry"`) to the leaf prior that produces it. `_prior_leaves: list[DensityPrior]` is the flat list of leaf nodes for optimizer construction.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def test_prior_leaf_by_key_populated_with_composite(fake_triangle_mesh: None) -> None:
    import warnings
    from sa_preisach.priors import CompositeDensityPrior, DiagonalDensityPrior, SymmetryDensityPrior
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior = CompositeDensityPrior(
            DiagonalDensityPrior(weight=1.0),
            SymmetryDensityPrior(weight=1.0),
        )
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        density_prior=prior,
    )
    assert "diagonal" in model._prior_leaf_by_key
    assert "symmetry" in model._prior_leaf_by_key
    assert len(model._prior_leaves) == 2


def test_prior_leaf_by_key_empty_without_prior(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
    )
    assert model._prior_leaf_by_key == {}
    assert model._prior_leaves == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_prior_leaf_by_key_populated_with_composite tests/test_encoder_decoder_preisach_nn.py::test_prior_leaf_by_key_empty_without_prior -v
```

Expected: FAIL — `_prior_leaves` not a list attribute.

- [ ] **Step 3: Replace `_inject_density_net` with extended version**

In `sa_preisach/models/_encoder_decoder_preisach_nn.py`, replace the `_inject_density_net` method and add the `_prior_leaf_by_key` / `_prior_leaves` init lines:

In `__init__`, replace:
```python
        self.model.density_prior = density_prior
        if density_prior is not None:
            self._inject_density_net(density_prior)
```

With:
```python
        self.model.density_prior = density_prior
        self._prior_leaves: list[DensityPrior] = []
        self._prior_leaf_by_key: dict[str, DensityPrior] = {}
        if density_prior is not None:
            self._collect_prior_leaves(density_prior)
```

Replace the `_inject_density_net` method:

```python
    def _collect_prior_leaves(self, prior: DensityPrior) -> None:
        from ..priors import CompositeDensityPrior, SymmetryDensityPrior

        if isinstance(prior, CompositeDensityPrior):
            for p in prior.priors:
                self._collect_prior_leaves(p)
        else:
            if isinstance(prior, SymmetryDensityPrior):
                prior.density_net = self.model.density_from_mesh
            # Discover the key this leaf produces by doing a dummy forward pass
            # on a tiny mesh so we can map key → leaf for weight application.
            with torch.no_grad():
                dummy_mesh = self.model.base_mesh.unsqueeze(0)[:, :1, :]
                dummy_density = torch.ones(1, 1)
                try:
                    sample = prior(dummy_mesh, dummy_density)
                except Exception:
                    sample = {}
            for k in sample:
                self._prior_leaf_by_key[k] = prior
            self._prior_leaves.append(prior)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_prior_leaf_by_key_populated_with_composite tests/test_encoder_decoder_preisach_nn.py::test_prior_leaf_by_key_empty_without_prior -v
```

Expected: both PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "refactor(model): replace _inject_density_net with _collect_prior_leaves, populate leaf map"
```

---

### Task 5: Add SA-PINN hyperparameters and `adaptive_weights` module to `EncoderDecoderPreisachNN`

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def test_adaptive_loss_weights_module_created_when_enabled(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    from sa_preisach.models._adaptive_loss_weights import AdaptiveLossWeights
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        adaptive_loss_weights=True,
        aux_loss_weight=2.0,
        saturation_reg_weight=0.5,
    )
    assert model.adaptive_weights is not None
    assert isinstance(model.adaptive_weights, AdaptiveLossWeights)
    assert torch.isclose(model.adaptive_weights.log_aux.exp(), torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(model.adaptive_weights.log_sat.exp(), torch.tensor(0.5), atol=1e-5)


def test_adaptive_loss_weights_none_when_disabled(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        adaptive_loss_weights=False,
    )
    assert model.adaptive_weights is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_module_created_when_enabled tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_none_when_disabled -v
```

Expected: FAIL — `adaptive_loss_weights` not a valid kwarg.

- [ ] **Step 3: Add hparams and `adaptive_weights` to `EncoderDecoderPreisachNN.__init__`**

Add these parameters to `EncoderDecoderPreisachNN.__init__` signature (after `fit_scale_offset`):

```python
        adaptive_loss_weights: bool = False,
        adaptive_loss_start: typing.Literal["all_phases", "phase2_plus"] = "phase2_plus",
        lr_adaptive: float = 1e-3,
```

Inside `__init__`, after the `_collect_prior_leaves` block, add:

```python
        from ._adaptive_loss_weights import AdaptiveLossWeights

        self.adaptive_weights: AdaptiveLossWeights | None = (
            AdaptiveLossWeights(
                aux_loss_weight=aux_loss_weight,
                saturation_reg_weight=saturation_reg_weight,
            )
            if adaptive_loss_weights
            else None
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_module_created_when_enabled tests/test_encoder_decoder_preisach_nn.py::test_adaptive_loss_weights_none_when_disabled -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "feat(model): add adaptive_loss_weights hparams and AdaptiveLossWeights module registration"
```

---

### Task 6: Update `optimizer_step` and `configure_optimizers` for the adaptive optimizer

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

- [ ] **Step 1: Write failing test for fourth optimizer**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def test_configure_optimizers_returns_four_when_adaptive(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        adaptive_loss_weights=True,
    )
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 4
    assert len(schedulers) == 4


def test_configure_optimizers_returns_two_when_not_adaptive(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        adaptive_loss_weights=False,
    )
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 2
    assert len(schedulers) == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_configure_optimizers_returns_four_when_adaptive tests/test_encoder_decoder_preisach_nn.py::test_configure_optimizers_returns_two_when_not_adaptive -v
```

Expected: FAIL.

- [ ] **Step 3: Update `optimizer_step` signature and logic**

Replace the existing `optimizer_step` function (lines ~124–154) with:

```python
def optimizer_step(  # noqa: PLR0913
    step: int,
    phase1_end: int,
    phase2_end: int,
    optimizer_encoder: torch.optim.Optimizer,
    optimizer_density: torch.optim.Optimizer,
    optimizer_scale: torch.optim.Optimizer | None,
    optimizer_adaptive: torch.optim.Optimizer | None,
    adaptive_loss_start: typing.Literal["all_phases", "phase2_plus"],
    clip_fn: typing.Callable[[torch.optim.Optimizer], None],
) -> None:
    adaptive_active = optimizer_adaptive is not None and (
        adaptive_loss_start == "all_phases" or step >= phase1_end
    )

    if step < phase1_end:
        clip_fn(optimizer_encoder)
        optimizer_encoder.step()
    elif step < phase2_end:
        clip_fn(optimizer_encoder)
        clip_fn(optimizer_density)
        optimizer_encoder.step()
        optimizer_density.step()
        if optimizer_scale is not None:
            clip_fn(optimizer_scale)
            optimizer_scale.step()
    else:
        step_offset = step - phase2_end
        if step_offset % 2 == 0:
            clip_fn(optimizer_encoder)
            optimizer_encoder.step()
        else:
            clip_fn(optimizer_density)
            optimizer_density.step()
            if optimizer_scale is not None:
                clip_fn(optimizer_scale)
                optimizer_scale.step()

    if adaptive_active:
        adaptive_params = list(optimizer_adaptive.param_groups[0]["params"])
        for p in adaptive_params:
            if p.grad is not None:
                p.grad.neg_()
        optimizer_adaptive.step()
```

- [ ] **Step 4: Add fourth optimizer to `configure_optimizers`**

At the end of `configure_optimizers`, before `return optimizers, schedulers`, add:

```python
        if self.hparams["adaptive_loss_weights"]:
            adaptive_params = list(self.adaptive_weights.parameters()) + [
                leaf.log_weight for leaf in self._prior_leaves
            ]
            optimizer_adaptive = torch.optim.AdamW(
                adaptive_params,
                lr=self.hparams["lr_adaptive"],
                weight_decay=0.0,
            )
            scheduler_adaptive = torch.optim.lr_scheduler.StepLR(
                optimizer_adaptive,
                step_size=self.hparams["lr_step_interval"],
                gamma=self.hparams["lr_gamma"],
            )
            optimizers.append(optimizer_adaptive)
            schedulers.append(scheduler_adaptive)
```

- [ ] **Step 5: Update `training_step` to retrieve and pass `optimizer_adaptive`**

In `training_step`, replace:

```python
        optimizer_scale = optimizers[2] if self.hparams["fit_scale_offset"] else None
```

With:

```python
        optimizer_scale = optimizers[2] if self.hparams["fit_scale_offset"] else None
        optimizer_adaptive = (
            optimizers[3] if self.hparams["adaptive_loss_weights"]
            else optimizers[2] if (self.hparams["fit_scale_offset"] and self.hparams["adaptive_loss_weights"])
            else None
        )
```

Wait — the optimizer index depends on whether `fit_scale_offset` is also on. The optimizer list order is:
- `[0]` encoder
- `[1]` density
- `[2]` scale (if `fit_scale_offset`)
- `[2 or 3]` adaptive (if `adaptive_loss_weights`)

So use:

```python
        optimizer_scale = optimizers[2] if self.hparams["fit_scale_offset"] else None
        _adaptive_idx = (3 if self.hparams["fit_scale_offset"] else 2)
        optimizer_adaptive = (
            optimizers[_adaptive_idx] if self.hparams["adaptive_loss_weights"] else None
        )
```

Also update the `optimizer_step` call to pass the new arguments:

```python
        optimizer_step(
            step=step,
            phase1_end=phase1_end,
            phase2_end=phase2_end,
            optimizer_encoder=optimizer_encoder,
            optimizer_density=optimizer_density,
            optimizer_scale=optimizer_scale,
            optimizer_adaptive=optimizer_adaptive,
            adaptive_loss_start=self.hparams["adaptive_loss_start"],
            clip_fn=clip_fn,
        )
```

Also update `training_step` to zero the adaptive optimizer before backward:

```python
        if optimizer_adaptive is not None:
            optimizer_adaptive.zero_grad()
```

And update the scheduler step block in `training_step` to also step the adaptive scheduler when active. After the existing scheduler block:

```python
        if self.trainer.is_last_batch:
            schedulers = self.lr_schedulers()
            schedulers[0].step()  # encoder
            if step >= phase1_end:
                schedulers[1].step()  # density
                if self.hparams["fit_scale_offset"]:
                    schedulers[2].step()  # scale/offset
            if self.hparams["adaptive_loss_weights"]:
                _adaptive_sched_idx = 3 if self.hparams["fit_scale_offset"] else 2
                if (
                    self.hparams["adaptive_loss_start"] == "all_phases"
                    or step >= phase1_end
                ):
                    schedulers[_adaptive_sched_idx].step()
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_configure_optimizers_returns_four_when_adaptive tests/test_encoder_decoder_preisach_nn.py::test_configure_optimizers_returns_two_when_not_adaptive -v
```

Expected: both PASS.

- [ ] **Step 7: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "feat(model): add adaptive optimizer with negated-gradient stepping, phase gating"
```

---

### Task 7: Update `common_step` loss computation for SA-PINN mode

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

In SA-PINN mode, `common_step` must compute weighted loss using `exp(log_w)` per term and also expose the unweighted combined loss. In static mode, behaviour is unchanged.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def _make_model_and_batch(
    adaptive: bool = False,
    fake_mesh: bool = True,
) -> tuple[EncoderDecoderPreisachNN, dict]:
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        adaptive_loss_weights=adaptive,
        encoder_fit_steps=0,
        density_fit_steps=0,
    )
    batch = {
        "encoder_input": torch.rand(2, 5, 2),
        "decoder_input": torch.rand(2, 4, 1),
        "target": torch.rand(2, 4, 1),
    }
    return model, batch


def test_common_step_has_loss_unweighted_when_adaptive(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model, batch = _make_model_and_batch(adaptive=True)
    model.eval()
    out = model.common_step(batch, 0)
    assert "loss_unweighted" in out


def test_common_step_no_loss_unweighted_when_not_adaptive(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model, batch = _make_model_and_batch(adaptive=False)
    model.eval()
    out = model.common_step(batch, 0)
    assert "loss_unweighted" not in out
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_common_step_has_loss_unweighted_when_adaptive tests/test_encoder_decoder_preisach_nn.py::test_common_step_no_loss_unweighted_when_not_adaptive -v
```

Expected: FAIL — `loss_unweighted` not in output.

- [ ] **Step 3: Refactor loss computation in `common_step`**

Before the `if self.adaptive_weights is not None` block, insert shared pre-computations for `aux_loss` and `physics_loss`. Both are needed in the adaptive branch (which doesn't call the `phase1_loss`/`phase2_loss` helpers), and `aux_loss` is also returned by the helpers in the static branch — so computing it once here and reusing it keeps the code DRY.

Insert this block immediately after the `saturation_reg` computation and before the `prior_losses_raw` block:

```python
        # Pre-compute aux_loss: needed in both static and adaptive paths.
        # Phase 1 uses mock_density; phase 2+ uses learned density.
        density_for_aux = self.model.mock_density if step < phase1_end else density
        density_sum_aux = density_for_aux.sum(dim=-1)
        m_initial = (density_for_aux * initial_states).sum(dim=-1) / density_sum_aux
        aux_loss = mse_loss(m_initial, m_target)

        # Pre-compute physics_loss for phase 1 (stratified MSE against deterministic
        # hysteron targets). Zero in phase 2+ where seq_loss takes its role.
        if step < phase1_end:
            h_last = y0.unsqueeze(-1)
            alpha_ph = mesh_coords[..., 1]
            beta_ph = mesh_coords[..., 0]
            mask_pos = beta_ph < h_last
            mask_neg = alpha_ph > h_last
            loss_pos = (
                mse_loss(initial_states[mask_pos], torch.ones_like(initial_states[mask_pos]))
                if mask_pos.any()
                else torch.tensor(0.0, device=initial_states.device)
            )
            loss_neg = (
                mse_loss(initial_states[mask_neg], -torch.ones_like(initial_states[mask_neg]))
                if mask_neg.any()
                else torch.tensor(0.0, device=initial_states.device)
            )
            physics_loss: torch.Tensor = 0.5 * (loss_pos + loss_neg)
        else:
            physics_loss = torch.zeros(1, device=initial_states.device)
```

Then replace the existing `if step < phase1_end: out = phase1_loss(...) else: out = phase2_loss(...)` block and the `return` dict with:

```python
        if self.adaptive_weights is not None:
            # SA-PINN mode: apply learned log-weights, also compute unweighted sum
            w_seq = self.adaptive_weights.log_seq.exp()
            w_aux = self.adaptive_weights.log_aux.exp()
            w_sat = self.adaptive_weights.log_sat.exp()

            prior_losses_weighted: dict[str, torch.Tensor] = {
                k: v * self._prior_leaf_by_key[k].log_weight.exp()
                for k, v in prior_losses_raw.items()
            } if prior_losses_raw else {}
            prior_loss_weighted = (
                sum(prior_losses_weighted.values())  # type: ignore[arg-type]
                if prior_losses_weighted
                else torch.zeros(1, device=density.device)
            )
            raw_prior_sum = (
                sum(prior_losses_raw.values())  # type: ignore[arg-type]
                if prior_losses_raw
                else torch.zeros(1, device=density.device)
            )

            if step < phase1_end:
                loss_unweighted = physics_loss + aux_loss + saturation_reg + raw_prior_sum
                loss = (
                    w_seq * physics_loss
                    + w_aux * aux_loss
                    + w_sat * saturation_reg
                    + prior_loss_weighted
                )
            else:
                seq_loss = mse_loss(y_hat, target_squeezed)
                loss_unweighted = seq_loss + aux_loss + saturation_reg + raw_prior_sum
                loss = (
                    w_seq * seq_loss
                    + w_aux * aux_loss
                    + w_sat * saturation_reg
                    + prior_loss_weighted
                )

            prior_losses = prior_losses_raw
            prior_loss = prior_loss_weighted

        else:
            # Static mode: existing behaviour via phase1_loss / phase2_loss helpers
            prior_losses = {
                k: v * self._prior_leaf_by_key[k].weight
                for k, v in prior_losses_raw.items()
            } if prior_losses_raw else {}
            prior_loss = (
                sum(prior_losses.values())  # type: ignore[arg-type]
                if prior_losses
                else torch.zeros(1, device=density.device)
            )

            if step < phase1_end:
                _out = phase1_loss(
                    initial_states=initial_states,
                    y0=y0,
                    mesh_coords=mesh_coords,
                    density=self.model.mock_density,
                    m_target=m_target,
                    saturation_reg=saturation_reg,
                    prior_loss=prior_loss,
                    aux_loss_weight=self.hparams["aux_loss_weight"],
                    saturation_reg_weight=self.hparams["saturation_reg_weight"],
                )
            else:
                _out = phase2_loss(
                    y_hat=y_hat,
                    target_squeezed=target_squeezed,
                    density=density,
                    initial_states=initial_states,
                    m_target=m_target,
                    saturation_reg=saturation_reg,
                    prior_loss=prior_loss,
                    aux_loss_weight=self.hparams["aux_loss_weight"],
                    saturation_reg_weight=self.hparams["saturation_reg_weight"],
                )
            loss = _out["loss"]
            loss_unweighted = None

        with torch.no_grad():
            residuals = y_hat.detach() - target_squeezed.detach()
            mse = (residuals**2).mean()
            rmse = mse.sqrt()
            mae = residuals.abs().mean()

        result: dict[str, torch.Tensor] = {
            "loss": loss,
            "aux_loss": aux_loss.detach(),
            "physics_loss": physics_loss.detach(),
            "saturation_reg": saturation_reg.detach(),
            "prior_loss": prior_loss.detach(),
            "prior_losses": {k: v.detach() for k, v in prior_losses.items()},
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "y_hat": y_hat,
            "y": target_squeezed,
            "x": decoder_input.squeeze(-1),
            "density": density_out.detach().clone(),
            "initial_states": initial_states.detach().clone(),
            "mesh_coords": mesh_coords.detach().clone(),
        }
        if loss_unweighted is not None:
            result["loss_unweighted"] = loss_unweighted.detach()
        return result
```

Note: the existing `phase1_loss` and `phase2_loss` functions still exist and are used in the static path. They compute `aux_loss` internally but we no longer extract it from their return dict — `aux_loss` is pre-computed above and used directly in `result`. The helpers' internal `aux_loss` is now redundant in the static path but harmless. If desired, the helpers can be simplified in a follow-up.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_common_step_has_loss_unweighted_when_adaptive tests/test_encoder_decoder_preisach_nn.py::test_common_step_no_loss_unweighted_when_not_adaptive -v
```

Expected: both PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "feat(model): SA-PINN adaptive loss weighting in common_step"
```

---

### Task 8: Add logging for adaptive weights and `loss_unweighted`

**Files:**
- Modify: `sa_preisach/models/_encoder_decoder_preisach_nn.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_encoder_decoder_preisach_nn.py`:

```python
def test_common_step_returns_adaptive_weight_values(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model, batch = _make_model_and_batch(adaptive=True)
    model.eval()
    out = model.common_step(batch, 0)
    assert "adaptive_weights" in out
    aw = out["adaptive_weights"]
    assert "seq" in aw
    assert "aux" in aw
    assert "sat" in aw
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_common_step_returns_adaptive_weight_values -v
```

Expected: FAIL.

- [ ] **Step 3: Add `adaptive_weights` dict to `common_step` return**

In the `result` dict construction, add:

```python
        if self.adaptive_weights is not None:
            result["adaptive_weights"] = {
                "seq": self.adaptive_weights.log_seq.exp().detach(),
                "aux": self.adaptive_weights.log_aux.exp().detach(),
                "sat": self.adaptive_weights.log_sat.exp().detach(),
                **{
                    f"prior/{k}": leaf.log_weight.exp().detach()
                    for k, leaf in self._prior_leaf_by_key.items()
                },
            }
```

- [ ] **Step 4: Add logging calls in `training_step` and `validation_step`**

In `training_step`, after the existing `for tag, key in {...}` log block, add:

```python
        if self.hparams["adaptive_loss_weights"]:
            self.log("train/loss_unweighted", out["loss_unweighted"], on_step=True, on_epoch=False)
            for k, v in out["adaptive_weights"].items():
                self.log(f"train/adaptive_weight/{k}", v, on_step=True, on_epoch=False)
```

In `validation_step`, after the existing log block, add:

```python
        if self.hparams["adaptive_loss_weights"]:
            self.log("validation/loss_unweighted", out["loss_unweighted"], on_step=False, on_epoch=True)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_encoder_decoder_preisach_nn.py::test_common_step_returns_adaptive_weight_values -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add sa_preisach/models/_encoder_decoder_preisach_nn.py tests/test_encoder_decoder_preisach_nn.py
git commit -m "feat(model): log adaptive weights and loss_unweighted"
```

---

### Task 9: Update `docs/source/encoder_decoder_preisach.rst`

**Files:**
- Modify: `docs/source/encoder_decoder_preisach.rst`

- [ ] **Step 1: Add new section and update combined loss block**

After the "Combined loss" section (around line 189), insert a new section:

```rst
Self-adaptive loss weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enabled with ``adaptive_loss_weights=True``. Each loss term is assigned a learned
log-weight parameter ``λᵢ`` (an ``nn.Parameter``), and the combined loss becomes:

.. math::

    \mathcal{L} = \sum_i \exp(\lambda_i) \cdot L_i

The model parameters are updated to **minimize** this weighted loss. The weight
parameters ``λᵢ`` are updated to **maximize** it (gradients are negated before
the adaptive optimizer steps). This pushes weight mass toward whichever term the
model is currently failing at, providing automatic loss balancing without manual
tuning of scalar weights.

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
```

Also update the "Combined loss" block to reflect both modes:

```rst
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

    where λ_* are learned parameters updated by maximizing the weighted loss
    (negated gradient step).
```

- [ ] **Step 2: Verify the docs build without errors**

```bash
cd docs && make html 2>&1 | tail -20
```

Expected: no errors or warnings related to the new section.

- [ ] **Step 3: Commit**

```bash
git add docs/source/encoder_decoder_preisach.rst
git commit -m "docs: add SA-PINN adaptive loss weighting section and update combined loss"
```

---

### Task 10: Final integration check

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 2: Run pre-commit**

```bash
pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Smoke test instantiation with all options**

```python
# Run interactively: python -c "..."
import torch
from sa_preisach.models import EncoderDecoderPreisachNN
from sa_preisach.nn import PreisachLSTMEncoder
from sa_preisach.priors import CompositeDensityPrior, DiagonalDensityPrior, SymmetryDensityPrior
import warnings

encoder = PreisachLSTMEncoder(num_features=2, hidden_dim=8, num_layers=1, dropout=0.0)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    prior = CompositeDensityPrior(DiagonalDensityPrior(1.0), SymmetryDensityPrior(1.0))

model = EncoderDecoderPreisachNN(
    mesh_scale=0.3,
    hidden_dim=8,
    num_layers=2,
    compile_model=False,
    encoder=encoder,
    density_prior=prior,
    adaptive_loss_weights=True,
    adaptive_loss_start='phase2_plus',
    lr_adaptive=1e-3,
)
print('adaptive_weights:', model.adaptive_weights)
print('prior_leaves:', model._prior_leaves)
print('prior_leaf_by_key:', list(model._prior_leaf_by_key.keys()))
optimizers, schedulers = model.configure_optimizers()
print('num optimizers:', len(optimizers))
```

Expected output:
```
adaptive_weights: AdaptiveLossWeights(...)
prior_leaves: [DiagonalDensityPrior(...), SymmetryDensityPrior(...)]
prior_leaf_by_key: ['diagonal', 'symmetry']
num optimizers: 4
```

- [ ] **Step 4: Commit spec and plan**

```bash
git add docs/superpowers/specs/2026-04-21-sa-pinn-loss-weights-design.md docs/superpowers/plans/2026-04-21-sa-pinn-loss-weights.md
git commit -m "docs: add SA-PINN design spec and implementation plan"
```
