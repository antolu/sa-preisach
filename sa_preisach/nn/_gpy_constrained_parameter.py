from __future__ import annotations

import typing

import gpytorch
import gpytorch.constraints
import torch


class GPyConstrainedParameter(gpytorch.Module):
    def __init__(
        self,
        data: torch.Tensor,
        constraint: gpytorch.constraints.Interval,
        *,
        requires_grad: bool = True,
        **kwargs: typing.Any,
    ):
        super().__init__()
        self.constraint = constraint

        self.raw_parameter = torch.nn.Parameter(
            self.inverse(data), requires_grad=requires_grad, **kwargs
        )
        self.register_constraint("raw_parameter", constraint)

    def forward(self) -> torch.Tensor:
        return self.constraint.transform(self.raw_parameter)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.constraint.inverse_transform(x)

    @property
    def data(self) -> torch.Tensor:
        return self.raw_parameter

    @property
    def value(self) -> torch.Tensor:
        return self.forward()
