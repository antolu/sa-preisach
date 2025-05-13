from __future__ import annotations

import torch
from transformertf.nn import get_activation, VALID_ACTIVATIONS


class ResNetMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        activation: VALID_ACTIVATIONS = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout

        # if isinstance(hidden_dim, int):
        #     hidden_dim = (hidden_dim,) * 3

        layers = []
        in_features = input_dim
        for dim in [hidden_dim] * num_layers:
            layers.append(torch.nn.Linear(in_features, dim))
            layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(get_activation(activation))
            layers.append(torch.nn.Dropout(dropout))
            in_features = dim

        self.residual_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(0, len(self.residual_layers), 4):
            residual = x if i != 0 else 0.0
            x = self.residual_layers[i](x)
            x = self.residual_layers[i + 1](x)
            x = self.residual_layers[i + 2](x)
            x = self.residual_layers[i + 3](x)
            # x += residual  # Add residual connection
            x = x + residual  # noqa: PLR6104
        return self.output_layer(x)
