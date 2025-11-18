from __future__ import annotations

import torch
from transformertf.nn import GatedResidualNetwork as GRN  # noqa: N817

from ._smooth_switch import SmoothSwitch


class PreisachLSTMEncoder(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        tanh_temp: float = 1e-2,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.mesh_embedding = GRN(
            input_dim=3,
            d_hidden=hidden_dim,
            output_dim=hidden_dim,
            context_dim=None,
            dropout=dropout,
            activation="lrelu",
        )

        self.state_grn = GRN(
            input_dim=hidden_dim,
            d_hidden=hidden_dim,
            output_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout,
            activation="lrelu",
        )

        self.state_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            SmoothSwitch(temp=tanh_temp),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mesh_coords: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _seq_len, _ = sequence.shape
        batch_size_mesh, n_mesh_points, _ = mesh_coords.shape

        assert batch_size == batch_size_mesh

        lstm_out, (h_n, _c_n) = self.lstm(sequence)

        last_hidden = h_n[-1]

        mesh_embedded = self.mesh_embedding(mesh_coords)

        last_hidden_expanded = last_hidden.unsqueeze(1).expand(-1, n_mesh_points, -1)

        state_features = self.state_grn(mesh_embedded, context=last_hidden_expanded)
        initial_states = self.state_head(state_features)

        return initial_states.squeeze(-1)
