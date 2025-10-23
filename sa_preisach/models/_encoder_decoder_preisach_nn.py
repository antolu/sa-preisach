from __future__ import annotations

import itertools
import typing

import einops
import gpytorch.constraints
import numpy as np
import torch
from transformertf.data import EncoderDecoderTargetSample
from transformertf.models._base_transformer import create_mask
from transformertf.nn.functional import mse_loss

from ..nn import GPyConstrainedParameter, PreisachTransformerEncoder, ResNetMLP
from ..utils import (
    create_triangle_mesh,
    get_states,
    make_mesh_size_function,
)
from ._base import BaseModule

CPU_DEVICE = torch.device("cpu")


class EncoderDecoderPreisachNNModel(torch.nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        mesh_size: float,
        num_past_features: int = 2,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        hidden_dim: int,
        num_layers: int = 3,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        normalized_density: bool = True,
        mesh_density_function: typing.Literal["constant", "default", "exponential"]
        | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] = "default",
        encoder_dropout: float = 0.1,
        mesh_perturbation_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.mesh_perturbation_std = mesh_perturbation_std

        # Create base mesh (will be expanded per batch)
        base_mesh = torch.from_numpy(
            create_triangle_mesh(
                mesh_size,
                mesh_density_function=mesh_density_function
                if callable(mesh_density_function)
                else make_mesh_size_function(mesh_density_function),
            )
        ).float()
        self.register_buffer("base_mesh", base_mesh)
        self.n_mesh_points = self.base_mesh.shape[0]

        # Transformer encoder for generating initial states
        self.encoder = PreisachTransformerEncoder(
            num_features=num_past_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=encoder_dropout,
            dim_feedforward=hidden_dim,
        )

        # Density network (same as original model)
        self.density = ResNetMLP(
            input_dim=2,
            output_dim=1,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation="relu",
            dropout=0.1,
        )
        self.density_activation = (
            torch.nn.Sigmoid() if normalized_density else torch.nn.LeakyReLU()
        )

        # Scale and offset parameters (same as original model)
        self.m_scale = GPyConstrainedParameter(
            torch.tensor(1.0),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=True,
        )
        self.m_offset = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*offset_bounds),
        )

    def _perturb_mesh(
        self, mesh_coords: torch.Tensor, training: bool
    ) -> torch.Tensor:
        """
        Perturb mesh coordinates during training for density network regularization.

        Parameters
        ----------
        mesh_coords : torch.Tensor
            Mesh coordinates of shape [batch_size, n_mesh_points, 2]
            where [..., 0] is beta and [..., 1] is alpha
        training : bool
            Whether the model is in training mode

        Returns
        -------
        torch.Tensor
            Perturbed mesh coordinates with shape [batch_size, n_mesh_points, 2]
            satisfying 0 <= alpha <= beta <= 1
        """
        if not training or self.mesh_perturbation_std == 0.0:
            return mesh_coords

        # Sample Gaussian noise for each (batch, mesh_point, coordinate)
        noise = torch.randn_like(mesh_coords) * self.mesh_perturbation_std

        # Add noise and clip to [0, 1]
        perturbed = torch.clamp(mesh_coords + noise, 0.0, 1.0)

        # Enforce beta >= alpha constraint
        # mesh_coords[..., 0] is beta, mesh_coords[..., 1] is alpha
        beta = perturbed[..., 0]
        alpha = perturbed[..., 1]

        # Where beta < alpha, set beta = alpha
        beta = torch.maximum(beta, alpha)

        # Stack back into [batch_size, n_mesh_points, 2] format
        perturbed = torch.stack([beta, alpha], dim=-1)

        return perturbed

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder-decoder Preisach model.

        Parameters
        ----------
        encoder_input : torch.Tensor
            Historical H/B sequences of shape [batch_size, ctxt_seq_len, sequence_features]
        decoder_input : torch.Tensor
            Current H sequences of shape [batch_size, tgt_seq_len, 1]
        encoder_mask : torch.Tensor, optional
            Attention mask for encoder sequences of shape [batch_size, ctxt_seq_len]
        y0 : torch.Tensor, optional
            Initial field values of shape [batch_size, 1]. If None, uses first decoder input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Magnetization predictions [batch_size, tgt_seq_len] and density weights [batch_size, n_mesh_points]
        """
        batch_size = encoder_input.shape[0]
        decoder_input.shape[1]

        # Expand mesh coordinates for batch
        mesh_coords = self.base_mesh.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, n_mesh_points, 2]

        # Perturb mesh during training for density network regularization
        mesh_coords = self._perturb_mesh(mesh_coords, self.training)

        # Encode initial hysteron states from historical sequences
        initial_states = self.encoder(
            sequence=encoder_input,
            mesh_coords=mesh_coords,
            sequence_mask=encoder_mask,
        )  # [batch_size, n_mesh_points]

        # Get initial field values - ensure always has batch dimension
        y0 = encoder_input[:, -1, -1] if y0 is None else y0.squeeze(-1)  # [batch_size]

        # Ensure y0 always has batch dimension for vmap compatibility
        if y0.dim() == 0:  # scalar case when batch_size=1
            y0 = y0.unsqueeze(0)  # [1]

        # Compute hysteron states for decoder sequence
        h = decoder_input.squeeze(-1)  # [batch_size, tgt_seq_len]
        alpha = mesh_coords[:, :, 1]  # [batch_size, n_mesh_points]
        beta = mesh_coords[:, :, 0]  # [batch_size, n_mesh_points]

        # Move data to CPU for state computation (sequential operation)
        # This allows NN operations to run on GPU while state updates run on CPU
        h_cpu = h.cpu()
        alpha_cpu = alpha.cpu()
        beta_cpu = beta.cpu()
        initial_states_cpu = initial_states.cpu()
        y0_cpu = y0.cpu()

        # Process each batch element separately using get_states on CPU
        # Note: torch.vmap doesn't work here due to data-dependent control flow in get_states
        batch_states = [
            get_states(
                h=h_cpu[b],  # [tgt_seq_len]
                alpha=alpha_cpu[b],  # [n_mesh_points]
                beta=beta_cpu[b],  # [n_mesh_points]
                current_state=initial_states_cpu[b],  # [n_mesh_points]
                current_field=y0_cpu[b],  # scalar
                temp=temp,
                dtype=torch.float32,
                training=self.training,
            )
            for b in range(batch_size)
        ]
        states_cpu = torch.stack(
            batch_states, dim=0
        )  # [batch_size, tgt_seq_len, n_mesh_points]

        # Move states back to original device for density computation
        states = states_cpu.to(h.device)

        # Compute density weights
        density = self.density(self.base_mesh)  # [n_mesh_points, 1]
        density = self.density_activation(density)
        density = einops.rearrange(density, "n 1 -> 1 n")  # [1, n_mesh_points]
        density = density.expand(batch_size, -1)  # [batch_size, n_mesh_points]

        # Compute magnetization for each time step
        m = torch.sum(density.unsqueeze(1) * states, dim=-1) / torch.sum(
            density.unsqueeze(1), dim=-1
        )  # [batch_size, tgt_seq_len]

        # Apply scale and offset
        m_out = self.m_scale.value * m + self.m_offset.value

        return m_out, density


class EncoderDecoderPreisachNN(BaseModule):
    model: EncoderDecoderPreisachNNModel

    """
    Encoder-Decoder Preisach Neural Network model with transformer encoder.

    This model extends the original Preisach NN by using a transformer encoder to generate
    initial hysteron states from historical sequences, enabling batched processing and
    arbitrary mesh scaling.

    Parameters
    ----------
    mesh_scale : float
        The scale of the mesh for creating the Preisach model.
    num_past_features : int, optional
        Number of features in input sequences (typically 2 for H and B). Default is 2.
    d_model : int, optional
        Transformer model dimension. Default is 128.
    num_heads : int, optional
        Number of transformer attention heads. Default is 8.
    num_encoder_layers : int, optional
        Number of transformer encoder layers. Default is 4.
    hidden_dim : int
        The number of hidden units in the MLP used to model the Preisach density.
    num_layers : int, optional
        The number of layers in the MLP used to model the Preisach density. Default is 3.
    temp : float, optional
        The temperature parameter for the hysteron activation function (tanh). Default is 1e-3.
    lr : float, optional
        The learning rate for the main optimizer. Default is 1e-2.
    lr_scale : float, optional
        The learning rate for the scale and offset parameters. Default is 1e-3.
    lr_step_interval : int, optional
        The interval at which to step the learning rate scheduler. Default is 100.
    lr_gamma : float, optional
        The factor by which to scale the learning rate at each step. Default is 0.9.
    m_scale_bounds : tuple[float, float], optional
        The bounds for the scale parameter of the Preisach model. Default is (0.0, 10.0).
    offset_bounds : tuple[float, float], optional
        The bounds for the offset parameter of the Preisach model. Default is (-10.0, 10.0).
    normalized_density : bool, optional
        Whether to normalize the density function. Default is True.
    mesh_density_function : str or callable, optional
        The function to use for the mesh density. Default is "default".
    compile_model : bool, optional
        Whether to compile the model using torch.compile. Default is True.
    encoder_dropout : float, optional
        Dropout probability for transformer encoder. Default is 0.1.
    n_train_samples : int, optional
        The number of training samples. Default is 0.
    """

    def __init__(  # noqa: PLR0913
        self,
        mesh_scale: float,
        *,
        num_past_features: int = 2,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        hidden_dim: int,
        num_layers: int = 3,
        temp: float = 1e-3,
        lr: float = 1e-2,
        lr_scale: float = 1e-3,
        lr_step_interval: int = 100,
        lr_gamma: float = 0.9,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        normalized_density: bool = True,
        mesh_density_function: typing.Literal["constant", "default", "exponential"]
        | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] = "default",
        compile_model: bool = True,
        encoder_dropout: float = 0.1,
        mesh_perturbation_std: float = 0.01,
        n_train_samples: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = EncoderDecoderPreisachNNModel(
            mesh_size=mesh_scale,
            num_past_features=num_past_features,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            normalized_density=normalized_density,
            mesh_density_function=mesh_density_function,
            encoder_dropout=encoder_dropout,
            mesh_perturbation_std=mesh_perturbation_std,
        )

    def on_train_epoch_start(self) -> None:
        return

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            encoder_input=encoder_input,
            decoder_input=decoder_input[..., 0:1],
            encoder_mask=encoder_mask,
            y0=y0,
            temp=temp,
        )

    def common_step(
        self,
        batch: EncoderDecoderTargetSample[torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch

        encoder_input = batch["encoder_input"]  # [batch_size, ctxt_seq_len, features]
        decoder_input = batch["decoder_input"]  # [batch_size, tgt_seq_len, features]
        target = batch["target"]  # [batch_size, tgt_seq_len, 1]

        # Get sequence lengths and flatten if needed
        encoder_lengths = batch.get("encoder_lengths")
        if encoder_lengths is not None:
            encoder_lengths = encoder_lengths[..., 0]  # (B, 1) -> (B,)

            # Create simple padding mask using TransformerTF utility
            encoder_mask = create_mask(
                size=encoder_input.shape[1],
                lengths=encoder_lengths,
                alignment="left",
                inverse=True,  # True=valid positions, False=padded
            )
        else:
            encoder_mask = batch.get("encoder_mask")
            if encoder_mask is not None:
                encoder_mask = encoder_mask.bool()

        # Get initial field from first target value
        y0 = target[:, 0, 0]  # [batch_size]

        y_hat, density = self(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            encoder_mask=encoder_mask,
            y0=y0,
            temp=self.hparams["temp"],
        )

        # Compute loss
        target_squeezed = target.squeeze(-1)  # [batch_size, tgt_seq_len]
        loss = mse_loss(y_hat, target_squeezed)

        return {
            "loss": loss,
            "y_hat": y_hat,
            "y": target_squeezed,
            "x": decoder_input.squeeze(-1),
            "density": density.detach().clone(),
        }

    def training_step(
        self, batch: EncoderDecoderTargetSample[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.common_step(batch, batch_idx)
        loss = out["loss"]

        # Logging
        for tag, key in {
            "train/loss": "loss",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch: EncoderDecoderTargetSample[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.common_step(batch, batch_idx)

        for tag, key in {
            "validation/loss": "loss",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=False, on_epoch=True)

        return out

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 1e-4,
                },
                {
                    "params": self.model.density.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 1e-4,
                },
                {
                    "params": itertools.chain(
                        self.model.m_scale.parameters(),
                        self.model.m_offset.parameters(),
                    ),
                    "lr": self.hparams["lr_scale"],
                    "weight_decay": 0.0,
                },
            ],
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        return optimizer, scheduler
