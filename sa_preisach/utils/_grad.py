from __future__ import annotations

import torch


def set_requires_grad(
    param_or_module: torch.nn.Module | torch.nn.Parameter, *, flag: bool
) -> None:
    if isinstance(param_or_module, torch.nn.Module):
        for p in param_or_module.parameters():
            p.requires_grad = flag
    else:
        param_or_module.requires_grad = flag
