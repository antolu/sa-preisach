from __future__ import annotations

import typing

import numpy as np
import pytest
import torch

from sa_preisach.models import PreisachNN
from sa_preisach.models._preisach_nn import PreisachNNModel
from transformertf.nn.functional import mse_loss


@pytest.fixture
def fake_triangle_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock(
        mesh_scale: float,
        mesh_density_function: typing.Callable[..., np.ndarray] | None = None,
    ) -> np.ndarray:
        del mesh_scale, mesh_density_function
        return np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "sa_preisach.models._preisach_nn.create_triangle_mesh",
        _mock,
    )


# ---------------------------------------------------------------------------
# masked_mse
# ---------------------------------------------------------------------------


def test_mse_loss_no_mask() -> None:
    y_hat = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    loss = mse_loss(y_hat, target)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_mse_loss_with_mask_ignores_padding() -> None:
    y_hat = torch.zeros(2, 4)
    target = torch.zeros(2, 4)
    target[0, :2] = 1000.0
    target[1, :3] = 1000.0
    lengths = torch.tensor([2, 3])
    T = 4
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    loss = mse_loss(y_hat, target, mask=mask)
    assert torch.isclose(loss, torch.tensor(1000.0**2))


def test_mse_loss_full_mask_same_as_no_mask() -> None:
    y_hat = torch.rand(2, 5)
    target = torch.rand(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert torch.isclose(mse_loss(y_hat, target, mask=mask), mse_loss(y_hat, target))


# ---------------------------------------------------------------------------
# PreisachNNModel
# ---------------------------------------------------------------------------


def test_model_forward_output_shapes(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNNModel(mesh_size=0.5, hidden_dim=8, num_layers=2, mesh_perturbation_std=0.0)
    batch, seq_len = 2, 6
    h = torch.rand(batch, seq_len)
    b_out, density, m, initial_states, mesh_coords = model(h)
    assert b_out.shape == (batch, seq_len)
    assert density.shape == (batch, 3)
    assert m.shape == (batch, seq_len)
    assert initial_states.shape == (batch, 3)
    assert mesh_coords.shape == (batch, 3, 2)


def test_model_initial_states_all_negative_one(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNNModel(mesh_size=0.5, hidden_dim=8, num_layers=2, mesh_perturbation_std=0.0)
    h = torch.rand(2, 5)
    _, _, _, initial_states, _ = model(h)
    assert (initial_states == -1.0).all()


def test_model_initial_states_override(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNNModel(mesh_size=0.5, hidden_dim=8, num_layers=2, mesh_perturbation_std=0.0)
    h = torch.rand(2, 5)
    custom_states = torch.ones(2, 3) * 0.5
    _, _, _, initial_states, _ = model(h, initial_states=custom_states)
    assert torch.allclose(initial_states, custom_states)


def test_model_initial_states_wrong_shape_raises(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNNModel(mesh_size=0.5, hidden_dim=8, num_layers=2, mesh_perturbation_std=0.0)
    h = torch.rand(2, 5)
    bad_states = torch.ones(2, 99)
    with pytest.raises(ValueError, match="initial_states shape"):
        model(h, initial_states=bad_states)


def test_model_no_mesh_perturbation_in_eval(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNNModel(mesh_size=0.5, hidden_dim=8, num_layers=2, mesh_perturbation_std=0.1)
    model.eval()
    h = torch.rand(1, 4)
    _, _, _, _, mesh_coords = model(h)
    base = model.base_mesh.unsqueeze(0)
    assert torch.allclose(mesh_coords, base)


# ---------------------------------------------------------------------------
# PreisachNN (Lightning module)
# ---------------------------------------------------------------------------


def test_module_instantiation(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNN(mesh_scale=0.5, hidden_dim=8, num_layers=2, compile_model=False)
    assert model.model.n_mesh_points == 3


def test_module_configure_optimizers_one_by_default(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNN(mesh_scale=0.5, hidden_dim=8, num_layers=2, compile_model=False)
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1


def test_module_configure_optimizers_two_with_fit_scale_offset(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    model = PreisachNN(
        mesh_scale=0.5, hidden_dim=8, num_layers=2, compile_model=False, fit_scale_offset=True
    )
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 2
    assert len(schedulers) == 2


def _make_module_and_batch(
    fake_triangle_mesh: None,
    *,
    with_lengths: bool = True,
) -> tuple[PreisachNN, dict[str, torch.Tensor]]:
    del fake_triangle_mesh
    model = PreisachNN(
        mesh_scale=0.5, hidden_dim=8, num_layers=2, compile_model=False, mesh_perturbation_std=0.0
    )
    batch: dict[str, torch.Tensor] = {
        "input": torch.rand(2, 5, 1),
        "target": torch.rand(2, 5, 1),
    }
    if with_lengths:
        batch["lengths"] = torch.tensor([4, 5])
    return model, batch


def test_common_step_returns_expected_keys(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh)
    model.eval()
    out = model.common_step(batch, 0)
    for key in ("loss", "seq_loss", "prior_loss", "y_hat", "y", "x"):
        assert key in out, f"missing key: {key}"


def test_common_step_loss_is_scalar(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh)
    model.eval()
    out = model.common_step(batch, 0)
    assert out["loss"].ndim == 0


def test_common_step_without_lengths(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh, with_lengths=False)
    model.eval()
    out = model.common_step(batch, 0)
    assert "loss" in out
    assert out["loss"].ndim == 0


def test_common_step_y_hat_shape(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh)
    model.eval()
    out = model.common_step(batch, 0)
    assert out["y_hat"].shape == (2, 5)


def test_common_step_density_shape(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh)
    model.eval()
    out = model.common_step(batch, 0)
    assert out["density"].shape == (2, 3)


def test_common_step_prior_losses_empty_without_prior(fake_triangle_mesh: None) -> None:
    model, batch = _make_module_and_batch(fake_triangle_mesh)
    model.eval()
    out = model.common_step(batch, 0)
    assert out["prior_losses"] == {}
