from __future__ import annotations

import typing

import torch


class BinaryParameter(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        requires_grad: bool = True,
        temp: float = 1e-2,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()

        self.temp = temp
        self.raw_parameter = torch.nn.Parameter(
            data, requires_grad=requires_grad, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.tanh(x / self.temp)

        return torch.sign(x)

    @property
    def data(self) -> torch.Tensor:
        return self.raw_parameter

    @property
    def value(self) -> torch.Tensor:
        return self.forward(self.raw_parameter)
