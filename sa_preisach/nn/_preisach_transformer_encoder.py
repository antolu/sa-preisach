from __future__ import annotations

import math
import typing

import torch
from transformertf.nn import get_activation, VALID_ACTIVATIONS


class PreisachTransformerEncoder(torch.nn.Module):
    """
    Transformer encoder for generating initial hysteron states from historical sequences.
    
    This encoder uses cross-attention between historical H/B sequences and mesh coordinates
    to determine appropriate initial states for each hysteron in the Preisach plane.
    
    Parameters
    ----------
    sequence_features : int
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
        sequence_features: int,
        *,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
    ) -> None:
        super().__init__()
        
        self.sequence_features = sequence_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.sequence_embedding = torch.nn.Linear(sequence_features, d_model)
        self.mesh_embedding = torch.nn.Linear(2, d_model)  # (alpha, beta) coordinates
        
        # Positional encoding for sequences
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        
        # Cross-attention between sequences and mesh points
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection to initial hysteron states [-1, 1]
        self.output_projection = torch.nn.Sequential(
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
        batch_size, seq_len, _ = sequence.shape
        batch_size_mesh, n_mesh_points, _ = mesh_coords.shape
        
        assert batch_size == batch_size_mesh, "Batch sizes must match"
        
        # Embed sequences and add positional encoding
        seq_embedded = self.sequence_embedding(sequence)  # [batch_size, seq_len, d_model]
        seq_embedded = self.pos_encoding(seq_embedded)
        
        # Create attention mask for transformer if provided
        if sequence_mask is not None:
            # Convert sequence mask to transformer attention mask format
            # sequence_mask: [batch_size, seq_len] -> [batch_size * num_heads, seq_len, seq_len]
            attention_mask = sequence_mask.unsqueeze(1).expand(-1, seq_len, -1)
            attention_mask = attention_mask.masked_fill(~attention_mask, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)
        else:
            attention_mask = None
        
        # Apply transformer encoder to sequence
        seq_encoded = self.transformer(
            seq_embedded, 
            src_key_padding_mask=~sequence_mask if sequence_mask is not None else None
        )  # [batch_size, seq_len, d_model]
        
        # Embed mesh coordinates
        mesh_embedded = self.mesh_embedding(mesh_coords)  # [batch_size, n_mesh_points, d_model]
        
        # Cross-attention: mesh points attend to sequence context
        # Query: mesh points, Key/Value: sequence context
        mesh_attended, _ = self.cross_attention(
            query=mesh_embedded,  # [batch_size, n_mesh_points, d_model]
            key=seq_encoded,      # [batch_size, seq_len, d_model] 
            value=seq_encoded,    # [batch_size, seq_len, d_model]
            key_padding_mask=~sequence_mask if sequence_mask is not None else None,
        )  # [batch_size, n_mesh_points, d_model]
        
        # Apply layer norm and residual connection
        mesh_attended = self.layer_norm(mesh_attended + mesh_embedded)
        
        # Project to initial hysteron states
        initial_states = self.output_projection(mesh_attended)  # [batch_size, n_mesh_points, 1]
        initial_states = initial_states.squeeze(-1)  # [batch_size, n_mesh_points]
        
        return initial_states


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
        self.register_buffer('pe', pe)
        
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
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)