from __future__ import annotations

import math

import torch
from transformertf.nn import VALID_ACTIVATIONS, get_activation


class PreisachTransformerEncoder(torch.nn.Module):
    """
    Transformer encoder for generating initial hysteron states from historical sequences.

    This encoder uses cross-attention between historical H/B sequences and mesh coordinates
    to determine appropriate initial states for each hysteron in the Preisach plane.

    Parameters
    ----------
    num_features : int
        Number of features in the historical sequence (typically 2 for H and B)
    d_model : int, optional
        Model dimension for transformer. Default is 128.
    num_heads : int, optional
        Number of attention heads. Default is 8.
    num_layers : int, optional
        Number of transformer encoder layers. Default is 4.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    activation : VALID_ACTIVATIONS, optional
        Activation function. Default is "relu".
    """

    def __init__(
        self,
        num_features: int,
        *,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layers
        self.sequence_embedding = torch.nn.Linear(num_features, d_model)
        # Store d_model for later use
        self.d_model = d_model

        # Positional encoding for sequences
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        # Use provided dim_feedforward or default to d_model (much smaller than PyTorch default)
        ff_dim = dim_feedforward if dim_feedforward is not None else d_model

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Mesh coordinate embedding (alpha, beta) -> d_model
        self.mesh_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, d_model // 2),
            get_activation(activation),
            torch.nn.Linear(d_model // 2, d_model),
        )

        # Cross-attention: mesh coordinates attend to sequence context
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final projection to initial states
        self.state_projection = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, 1),
            torch.nn.Tanh(),  # Ensure output in [-1, 1] range
        )

        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(
        self,
        sequence: torch.Tensor,
        mesh_coords: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.

        Parameters
        ----------
        sequence : torch.Tensor
            Historical H/B sequence data of shape [batch_size, seq_len, sequence_features]
        mesh_coords : torch.Tensor
            Mesh coordinates (alpha, beta) of shape [batch_size, n_mesh_points, 2]
        sequence_mask : torch.Tensor, optional
            Attention mask for variable-length sequences of shape [batch_size, seq_len]

        Returns
        -------
        torch.Tensor
            Initial hysteron states of shape [batch_size, n_mesh_points]
        """
        batch_size, _seq_len, _ = sequence.shape
        batch_size_mesh, _n_mesh_points, _ = mesh_coords.shape

        assert batch_size == batch_size_mesh, "Batch sizes must match"

        # Embed sequences and add positional encoding
        seq_embedded = self.sequence_embedding(
            sequence
        )  # [batch_size, seq_len, d_model]
        seq_embedded = self.pos_encoding(seq_embedded)

        # Apply transformer encoder to sequence
        # Use src_key_padding_mask where True=ignore, False=attend
        # Our mask has True=valid, so we need to invert it
        seq_encoded = self.transformer(
            seq_embedded,
            src_key_padding_mask=~sequence_mask if sequence_mask is not None else None,
        )  # [batch_size, seq_len, d_model]

        # Embed mesh coordinates
        mesh_embedded = self.mesh_embedding(
            mesh_coords
        )  # [batch_size, n_mesh_points, d_model]

        # Cross-attention: mesh points attend to sequence context
        # Query: mesh coordinates, Key/Value: sequence context
        mesh_attended, _ = self.cross_attention(
            query=mesh_embedded,  # [batch_size, n_mesh_points, d_model]
            key=seq_encoded,  # [batch_size, seq_len, d_model]
            value=seq_encoded,  # [batch_size, seq_len, d_model]
            key_padding_mask=~sequence_mask if sequence_mask is not None else None,
        )  # [batch_size, n_mesh_points, d_model]

        # Apply layer norm with residual connection
        mesh_contextualized = self.layer_norm(mesh_attended + mesh_embedded)

        # Project to initial states
        initial_states = self.state_projection(
            mesh_contextualized
        )  # [batch_size, n_mesh_points, 1]
        return initial_states.squeeze(-1)  # [batch_size, n_mesh_points]


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for transformer inputs.

    Parameters
    ----------
    d_model : int
        Model dimension
    dropout : float, optional
        Dropout probability. Default is 0.1.
    max_len : int, optional
        Maximum sequence length. Default is 5000.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, d_model]

        Returns
        -------
        torch.Tensor
            Input with positional encoding added
        """
        x += self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)
