from __future__ import annotations

import torch

from ._base import DensityPrior


class EntropyDensityPrior(DensityPrior):
    """
    Encourages a spread-out density distribution by penalizing low entropy.

    Loss = -mean over batch of ( sum_i p_i * log(p_i + eps) )

    where p_i = mu_i / sum_j mu_j is the normalized density (treated as a
    probability distribution over mesh points). Minimizing the negative entropy
    maximizes entropy, pushing the model toward broader coercivity distributions
    rather than delta-like concentrations.

    Useful for materials with a wide distribution of switching fields, or as a
    regularizer to prevent the density from collapsing to a single point.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__(weight)
        self.eps = eps

    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        density_sum = density.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p = density / density_sum
        entropy = -(p * (p + self.eps).log()).sum(dim=-1).mean()
        return {"entropy": -entropy}
