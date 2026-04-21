from __future__ import annotations

import torch

from ._base import DensityPrior


class CentroidDensityPrior(DensityPrior):
    """
    Penalizes the density-weighted centroid being far from the origin along the diagonal.

    Loss = mean over batch of ( sum_i mu_i * (alpha_i + beta_i) / 2 / sum_i mu_i )

    Encourages mass near small (alpha, beta) values — hysterons that flip at low
    fields. Useful for soft magnetic materials where most hysteretic activity
    occurs near zero field.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    @staticmethod
    def forward(
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        alpha = mesh_coords[..., 1]
        beta = mesh_coords[..., 0]
        midpoint = (alpha + beta) / 2.0
        density_sum = density.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        centroid = ((density * midpoint).sum(dim=-1) / density_sum.squeeze(-1)).mean()
        return {"centroid": centroid}
