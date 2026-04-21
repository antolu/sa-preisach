from __future__ import annotations

import torch

from ._base import DensityPrior


class CentroidDensityPrior(DensityPrior):
    """
    Penalizes the density-weighted centroid being far from a target (alpha, beta).

    Loss = mean over batch of (
        (sum_i mu_i * alpha_i / sum_i mu_i - target_alpha)^2
      + (sum_i mu_i * beta_i  / sum_i mu_i - target_beta)^2
    )

    With the [0, 1] normalized convention where 0.5 corresponds to zero field,
    use target_alpha=0.5, target_beta=0.5 to pull mass toward low-field hysterons.
    """

    def __init__(
        self,
        weight: float = 1.0,
        target_alpha: float = 0.5,
        target_beta: float = 0.5,
    ) -> None:
        super().__init__(weight)
        self.target_alpha = target_alpha
        self.target_beta = target_beta

    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        alpha = mesh_coords[..., 1]
        beta = mesh_coords[..., 0]
        density_sum = density.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        centroid_alpha = (density * alpha).sum(dim=-1) / density_sum.squeeze(-1)
        centroid_beta = (density * beta).sum(dim=-1) / density_sum.squeeze(-1)
        loss = (
            (centroid_alpha - self.target_alpha) ** 2
            + (centroid_beta - self.target_beta) ** 2
        ).mean()
        return {"centroid": loss}
