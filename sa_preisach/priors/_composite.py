from __future__ import annotations

import torch

from ._base import DensityPrior


class CompositeDensityPrior(DensityPrior):
    def __init__(self, *priors: DensityPrior) -> None:
        seen: set[type] = set()
        for p in priors:
            if type(p) in seen:
                msg = f"Duplicate prior type: {type(p).__name__}. Compose them externally instead."
                raise ValueError(msg)
            seen.add(type(p))
        self.priors = priors

    def __call__(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for prior in self.priors:
            for k, v in prior(mesh_coords, density).items():
                if k in out:
                    out[k] = out[k] + v
                else:
                    out[k] = v
        return out
