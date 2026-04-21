from __future__ import annotations

import torch

from ._base import DensityPrior


class BoundaryDensityPrior(DensityPrior):
    """
    Penalizes density mass near the triangle boundary.

    Loss = mean over batch of density-weighted mean of min(alpha, 1-beta, alpha-beta) inverted,
    i.e. how close each hysteron is to any of the three boundary edges:
      - alpha → 0  (top-left corner, near alpha=0)
      - beta  → 1  (right edge, near beta=1)
      - alpha - beta → 0  (diagonal edge)

    Loss = mean( sum_i mu_i * exp(-margin_i / sigma) / sum_i mu_i )

    where margin_i = min(alpha_i, 1 - beta_i, alpha_i - beta_i) is the distance
    to the nearest boundary, and sigma controls the softness of the penalty.
    """

    def __init__(self, weight: float = 1.0, sigma: float = 0.05) -> None:
        super().__init__(weight)
        self.sigma = sigma

    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        alpha = mesh_coords[..., 1]
        beta = mesh_coords[..., 0]
        margin = torch.stack([alpha, 1.0 - beta, alpha - beta], dim=-1).min(dim=-1).values
        boundary_proximity = torch.exp(-margin / self.sigma)
        density_sum = density.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        loss = ((density * boundary_proximity).sum(dim=-1) / density_sum.squeeze(-1)).mean()
        return {"boundary": loss}
