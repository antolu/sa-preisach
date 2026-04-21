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
