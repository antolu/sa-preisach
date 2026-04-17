from __future__ import annotations

import abc

import torch


class PreisachEncoder(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        sequence: torch.Tensor,
        mesh_features: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass
