from __future__ import annotations

import warnings

import torch

from ._base import DensityPrior


class SymmetryDensityPrior(DensityPrior):
    """
    Enforces mu(beta, alpha) = mu(1-alpha, 1-beta).

    Only valid for materials with symmetric major hysteresis loops (equal positive
    and negative saturation). Do not use for asymmetric materials.

    Uses density-weighted MSE so the penalty fires where mass actually exists,
    not diluted by near-zero regions.

    Parameters
    ----------
    density_fn : callable
        Function that evaluates the density network at arbitrary mesh coordinates.
        Signature: (mesh_coords: Tensor[batch, N, 2]) -> Tensor[batch, N].
        Typically ``model.density_from_mesh``.
    """

    def __init__(
        self,
        density_fn: torch.nn.Module,
    ) -> None:
        warnings.warn(
            "SymmetryDensityPrior assumes mu(b,a) = mu(1-a,1-b), which is only valid "
            "for materials with symmetric major hysteresis loops.",
            UserWarning,
            stacklevel=2,
        )
        self.density_fn = density_fn

    def __call__(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mirror_coords = torch.stack(
            [1.0 - mesh_coords[..., 1], 1.0 - mesh_coords[..., 0]], dim=-1
        )
        density_mirror = self.density_fn(mirror_coords).detach()
        weights = (density + density_mirror) / 2
        weights_sum = weights.sum(dim=-1).clamp(min=1e-8)
        loss = (
            (weights * (density - density_mirror) ** 2).sum(dim=-1) / weights_sum
        ).mean()
        return {"symmetry": loss}
