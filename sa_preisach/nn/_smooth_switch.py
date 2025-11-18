from __future__ import annotations

import torch


class SmoothSwitch(torch.nn.Module):
    """
    Smooth tanh-based switch with temperature control.

    During training, applies a smooth tanh transformation with temperature scaling.
    During evaluation, applies a discontinuous sign function.

    Parameters
    ----------
    temp : float
        Temperature parameter for smoothing. Lower values create sharper transitions.
    """

    def __init__(self, temp: float = 1e-2) -> None:
        super().__init__()
        self.temp = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.tanh(x / self.temp)
        return torch.sign(x)
