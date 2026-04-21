from __future__ import annotations

import math

import torch


class AdaptiveLossWeights(torch.nn.Module):
    def __init__(self, aux_loss_weight: float, saturation_reg_weight: float) -> None:
        super().__init__()
        self.log_seq = torch.nn.Parameter(torch.tensor(0.0))
        self.log_aux = torch.nn.Parameter(torch.tensor(math.log(aux_loss_weight)))
        self.log_sat = torch.nn.Parameter(torch.tensor(math.log(saturation_reg_weight)))
