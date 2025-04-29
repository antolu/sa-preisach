from __future__ import annotations

import typing

import torch


class ConstrainedParameter(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        min_: float = 0.0,
        max_: float = 1.0,
        eps: float = 1e-3,
        *,
        requires_grad: bool = True,
        **kwargs: typing.Any,
    ):
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.eps = eps

        data = data.clamp(min_ + eps, max_ - eps)
        self.raw_parameter = torch.nn.Parameter(
            self.inverse(data), requires_grad=requires_grad, **kwargs
        )

    def forward(self) -> torch.Tensor:
        unit_value = torch.nn.functional.softplus(self.raw_parameter)
        unit_value = unit_value / (1 + unit_value)

        # map to slightly wider interval
        stretched_value = (1 + 2 * self.eps) * unit_value - self.eps

        # finally clip to [0,1]
        stretched_value = torch.clamp(stretched_value, 0.0, 1.0)

        return stretched_value * (self.max_ - self.min_) + self.min_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Map from [min_, max_] to [0,1]
        unit_value = (x - self.min_) / (self.max_ - self.min_)

        # Reverse the stretch
        unit_value = (unit_value + self.eps) / (1 + 2 * self.eps)

        # Now unit_value ∈ (0,1), invert softplus-squash
        eps_ = 1e-5
        unit_value = torch.clamp(unit_value, eps_, 1 - eps_)

        softplus_value = unit_value / (1 - unit_value)
        return torch.log(torch.expm1(softplus_value))

    @property
    def data(self) -> torch.Tensor:
        return self.raw_parameter

    @property
    def value(self) -> torch.Tensor:
        return self.forward()
