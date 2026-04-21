from __future__ import annotations

import torch

from ._base import DensityPrior


class CompositeDensityPrior(DensityPrior):
    def __init__(self, *priors: DensityPrior) -> None:
        super().__init__(weight=1.0)
        self.priors = torch.nn.ModuleList(priors)

    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        key_counts: dict[str, int] = {}
        for prior in self.priors:
            for k, v in prior(mesh_coords, density).items():
                if k in out:
                    # rename both existing and new entry to avoid silent summation
                    count = key_counts.get(k, 1)
                    if count == 1:
                        out[f"{k}_0"] = out.pop(k)
                    out[f"{k}_{count}"] = v
                    key_counts[k] = count + 1
                else:
                    out[k] = v
                    key_counts[k] = 1
        return out
