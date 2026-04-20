from __future__ import annotations

import abc

import torch


class DensityPrior(torch.nn.Module, abc.ABC):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    @abc.abstractmethod
    def forward(
        self,
        mesh_coords: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mesh_coords : torch.Tensor
            Mesh coordinates [batch, N, 2] where [..., 0] is beta and [..., 1] is alpha.
        density : torch.Tensor
            Density values [batch, N].

        Returns
        -------
        dict[str, torch.Tensor]
            Named scalar loss terms. The model sums all values for the total prior
            loss and logs each key individually under ``train/prior/<key>``.
        """
