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

    The density evaluation function (including activation) is injected via
    ``density_net`` after construction — set to ``model.density_from_mesh``,
    not the raw MLP, so outputs are properly bounded in (0, 1).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        warnings.warn(
            "SymmetryDensityPrior assumes mu(b,a) = mu(1-a,1-b), which is only valid "
            "for materials with symmetric major hysteresis loops.",
            UserWarning,
            stacklevel=2,
        )
        self.density_net: torch.nn.Module | None = None

    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.density_net is None:
            msg = "SymmetryDensityPrior.density_net is not set. Register this prior on the model first."
            raise RuntimeError(msg)
        mirror_coords = 1.0 - mesh_coords
        with torch.no_grad():
            density_mirror = self.density_net(
                mirror_coords
            )  # [batch, N], activation already applied
        weights = (density + density_mirror) / 2
        weights_sum = weights.sum(dim=-1).clamp(min=1e-8)
        loss = (
            (weights * (density - density_mirror) ** 2).sum(dim=-1) / weights_sum
        ).mean()
        return {"symmetry": loss}
