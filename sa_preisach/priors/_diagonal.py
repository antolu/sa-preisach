from __future__ import annotations

import torch

from ._base import DensityPrior


class DiagonalDensityPrior(DensityPrior):
    """
    Penalizes density mass away from the alpha=beta diagonal.

    Loss = mean over batch of ( sum_i mu_i * (alpha_i - beta_i)^2 / sum_i mu_i )

    Density-weighted so sparse/concentrated distributions are penalized where
    mass actually exists, not diluted by near-zero regions.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def __call__(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        alpha = mesh_coords[..., 1]
        beta = mesh_coords[..., 0]
        dist_sq = (alpha - beta) ** 2
        density_sum = density.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        loss = ((density * dist_sq).sum(dim=-1) / density_sum.squeeze(-1)).mean()
        return {"diagonal": self.weight * loss}
