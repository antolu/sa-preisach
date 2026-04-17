from __future__ import annotations

import logging
import typing

import gpytorch.constraints
import numpy as np
import torch
from transformertf.data import EncoderDecoderTargetSample
from transformertf.models._base_transformer import create_mask
from transformertf.nn.functional import mse_loss

from ..nn import (
    GPyConstrainedParameter,
    PreisachEncoder,
    PreisachLSTMEncoder,
    ResNetMLP,
)
from ..utils import (
    create_triangle_mesh,
    get_states,
    make_mesh_size_function,
)
from ._base import BaseModule

log = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")
MIN_VARIANCE_THRESHOLD = 1e-6


class EncoderDecoderPreisachNNModel(torch.nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        mesh_size: float,
        encoder: PreisachEncoder,
        hidden_dim: int,
        num_layers: int = 3,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        normalized_density: bool = True,
        mesh_density_function: typing.Literal["constant", "default", "exponential"]
        | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] = "default",
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

        self.encoder = encoder

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
            torch.nn.Sigmoid() if normalized_density else torch.nn.Softplus()
        )

        # Separate scale parameters for H (applied field) and M (magnetization)
        # B = h_scale * H + m_scale * M + offset
        self.h_scale = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=False,
        )
        self.m_scale = GPyConstrainedParameter(
            torch.tensor(1.0),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=True,
        )
        self.m_offset = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*offset_bounds),
            requires_grad=True,
        )

    def _perturb_mesh(self, mesh_coords: torch.Tensor, training: bool) -> torch.Tensor:  # noqa: FBT001
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

        # Enforce beta <= alpha constraint
        # mesh_coords[..., 0] is beta, mesh_coords[..., 1] is alpha
        beta = perturbed[..., 0]
        alpha = perturbed[..., 1]

        # Where beta > alpha, set beta = alpha
        beta = torch.minimum(beta, alpha)

        # Stack back into [batch_size, n_mesh_points, 2] format
        return torch.stack([beta, alpha], dim=-1)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            B predictions, density weights, unscaled M, initial states, mesh coordinates
        """
        batch_size = encoder_input.shape[0]

        # Expand mesh coordinates for batch
        mesh_coords = self.get_batched_mesh_coords(
            batch_size
        )  # [batch_size, n_mesh_points, 2]

        # Perturb mesh during training for density network regularization
        mesh_coords = self._perturb_mesh(mesh_coords, self.training)

        # Compute density weights first
        density = self.density_from_mesh(
            mesh_coords, beta=None
        )  # [batch_size, n_mesh_points]

        # Concatenate density to mesh coordinates for encoder
        mesh_coords_with_density = torch.cat(
            [mesh_coords, density.unsqueeze(-1)], dim=-1
        )  # [batch_size, n_mesh_points, 3]

        # Encode initial hysteron states from historical sequences
        initial_states = self.encoder(
            sequence=encoder_input,
            mesh_features=mesh_coords_with_density,
            sequence_mask=encoder_mask,
        )  # [batch_size, n_mesh_points]

        # Get initial field values - ensure always has batch dimension
        y0 = encoder_input[:, -1, -1] if y0 is None else y0.squeeze(-1)  # [batch_size]

        # Ensure y0 always has batch dimension for vmap compatibility
        if y0.dim() == 0:  # scalar case when batch_size=1
            y0 = y0.unsqueeze(0)  # [1]

        # Compute hysteron states for decoder sequence
        h = decoder_input.squeeze(-1)  # [batch_size, tgt_seq_len]
        alpha = mesh_coords[..., 1]  # [batch_size, n_mesh_points]
        beta = mesh_coords[..., 0]  # [batch_size, n_mesh_points]

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

        # Move states back to original device
        states = states_cpu.to(h.device)

        # Normalize density once (constant across time)
        density_sum = density.sum(dim=-1, keepdim=True)  # [batch_size, 1]

        # Compute magnetization for each time step
        m = (
            torch.sum(density.unsqueeze(1) * states, dim=-1) / density_sum
        )  # [batch_size, tgt_seq_len]

        # Apply physics-informed scaling: B = h_scale * H + m_scale * M + offset
        b_out = self.h_scale.value * h + self.m_scale.value * m + self.m_offset.value

        return (
            b_out,
            density,
            m,
            initial_states,
            mesh_coords,
        )  # Return B, density, unscaled M, initial states, and mesh coords

    def get_batched_mesh_coords(self, batch_size: int) -> torch.Tensor:
        """
        Get batched mesh coordinates for the entire batch.

        Parameters
        ----------
        batch_size : int
            The size of the batch

        Returns
        -------
        torch.Tensor
            Batched mesh coordinates of shape [batch_size, n_mesh_points, 2]
        """
        return self.base_mesh.unsqueeze(0).expand(batch_size, -1, -1)

    def density_from_mesh(
        self, mesh_coords: torch.Tensor, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute density values from mesh coordinates.

        Parameters
        ----------
        mesh_coords : torch.Tensor
            Mesh coordinates of shape [batch_size, n_mesh_points, 2]

        Returns
        -------
        torch.Tensor
            Density values of shape [batch_size, n_mesh_points]
        """
        if beta is not None:  # assume mesh_coords is alpha
            mesh_coords = torch.cat(
                [beta.unsqueeze(-1), mesh_coords], dim=-1
            )  # [batch_size, n_mesh_points, 2]
        density = self.density(mesh_coords)  # [batch_size, n_mesh_points, 1]
        density = self.density_activation(density)
        return density.squeeze(-1)  # [batch_size, n_mesh_points]


class EncoderDecoderPreisachNN(BaseModule):
    model: EncoderDecoderPreisachNNModel

    def __init__(  # noqa: PLR0913
        self,
        mesh_scale: float,
        *,
        num_past_features: int = 2,
        encoder_hidden_dim: int | None = None,
        encoder_num_layers: int = 2,
        encoder_dropout: float = 0.1,
        encoder: PreisachEncoder | None = None,
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
        mesh_perturbation_std: float = 0.01,
        linear_fit_steps: int = 0,
        encoder_fit_steps: int = 0,
        density_fit_steps: int = 0,
        gradient_clip_val: float = 1.0,
        n_train_samples: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        encoder_hdim = (
            encoder_hidden_dim if encoder_hidden_dim is not None else hidden_dim
        )
        encoder_module = encoder or PreisachLSTMEncoder(
            num_features=num_past_features,
            hidden_dim=encoder_hdim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        self.model = EncoderDecoderPreisachNNModel(
            mesh_size=mesh_scale,
            encoder=encoder_module,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            normalized_density=normalized_density,
            mesh_density_function=mesh_density_function,
            mesh_perturbation_std=mesh_perturbation_std,
        )

        self._scale_offset_initialized = False
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        log.info(f"Number of mesh points: {self.model.n_mesh_points}")

    def _initialize_scale_offset(self, m: torch.Tensor, target: torch.Tensor) -> None:
        """
        Analytically initialize m_scale and m_offset using least squares fit.

        Given target = scale * m + offset, we solve for scale and offset:
        - scale = cov(target, m) / var(m)
        - offset = mean(target) - scale * mean(m)

        Parameters
        ----------
        m : torch.Tensor
            Raw magnetization output before scaling
        target : torch.Tensor
            Target magnetization values
        """
        with torch.no_grad():
            # Flatten to 1D for statistics
            m_flat = m.flatten()
            target_flat = target.flatten()

            # Compute means
            m_mean = m_flat.mean()
            target_mean = target_flat.mean()

            # Compute scale: cov(target, m) / var(m)
            m_centered = m_flat - m_mean
            target_centered = target_flat - target_mean

            covariance = (m_centered * target_centered).sum()
            variance = (m_centered**2).sum()

            # Avoid division by zero
            if variance > MIN_VARIANCE_THRESHOLD:
                scale = covariance / variance
                offset = target_mean - scale * m_mean

                # Clamp to valid bounds
                scale = torch.clamp(
                    scale,
                    torch.tensor(self.hparams["m_scale_bounds"][0]),
                    torch.tensor(self.hparams["m_scale_bounds"][1]),
                )
                offset = torch.clamp(
                    offset,
                    torch.tensor(self.hparams["offset_bounds"][0]),
                    torch.tensor(self.hparams["offset_bounds"][1]),
                )

                # Initialize parameters using inverse transform
                self.model.m_scale.raw_parameter.data = self.model.m_scale.inverse(
                    scale.to(self.model.m_scale.raw_parameter.device)
                )
                self.model.m_offset.raw_parameter.data = self.model.m_offset.inverse(
                    offset.to(self.model.m_offset.raw_parameter.device)
                )

                log.info(
                    f"Initialized m_scale={scale.item():.4f}, m_offset={offset.item():.4f}"
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Get initial field from last encoder value (y0=None lets model use encoder_input[:, -1, -1])
        y_hat, density, m_unscaled, initial_states, mesh_coords = self(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            encoder_mask=encoder_mask,
            y0=None,
            temp=self.hparams["temp"],
        )

        if self.training and not self._scale_offset_initialized:
            target_squeezed = target.squeeze(-1)
            self._initialize_scale_offset(m_unscaled, target_squeezed)
            self._scale_offset_initialized = True

        # Compute loss
        target_squeezed = target.squeeze(-1)  # [batch_size, tgt_seq_len]

        # During encoder-only fitting, only compute loss on first timestep
        step = self.global_step if self.training else float("inf")
        phase1_end = self.hparams["linear_fit_steps"]
        phase2_end = phase1_end + self.hparams["encoder_fit_steps"]

        if step >= phase1_end and step < phase2_end:
            loss = mse_loss(y_hat[:, 0], target_squeezed[:, 0])
        else:
            loss = mse_loss(y_hat, target_squeezed)

        return {
            "loss": loss,
            "y_hat": y_hat,
            "y": target_squeezed,
            "x": decoder_input.squeeze(-1),
            "density": density.detach().clone(),
            "initial_states": initial_states.detach().clone(),
            "mesh_coords": mesh_coords.detach().clone(),
        }

    def training_step(  # noqa: PLR0915, PLR0912
        self, batch: EncoderDecoderTargetSample[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        optimizer_encoder, optimizer_density, optimizer_offset, optimizer_m_scale = (
            self.optimizers()
        )

        out = self.common_step(batch, batch_idx)
        loss = out["loss"]

        step = self.global_step
        phase1_end = self.hparams["linear_fit_steps"]
        phase2_end = phase1_end + self.hparams["encoder_fit_steps"]
        phase3_end = phase2_end + self.hparams["density_fit_steps"]

        optimizer_encoder.zero_grad()
        optimizer_density.zero_grad()
        optimizer_offset.zero_grad()
        optimizer_m_scale.zero_grad()

        self.manual_backward(loss)

        if step < phase1_end:
            optimizer_m_scale.step()
            optimizer_offset.step()
        elif step < phase2_end:
            self.clip_gradients(
                optimizer_encoder, gradient_clip_val=self.hparams["gradient_clip_val"]
            )
            optimizer_encoder.step()
        elif step < phase3_end:
            optimizer_m_scale.step()
            optimizer_offset.step()
            self.clip_gradients(
                optimizer_density, gradient_clip_val=self.hparams["gradient_clip_val"]
            )
            optimizer_density.step()
        else:
            optimizer_m_scale.step()
            optimizer_offset.step()
            if step == phase3_end:
                for param_group in optimizer_m_scale.param_groups:
                    param_group["lr"] /= 10.0
                for param_group in optimizer_offset.param_groups:
                    param_group["lr"] /= 10.0

            step_offset = step - phase3_end
            cycle_position = step_offset % 3

            if cycle_position == 0:
                pass
            elif cycle_position == 1:
                self.clip_gradients(
                    optimizer_encoder,
                    gradient_clip_val=self.hparams["gradient_clip_val"],
                )
                optimizer_encoder.step()
            else:
                self.clip_gradients(
                    optimizer_density,
                    gradient_clip_val=self.hparams["gradient_clip_val"],
                )
                optimizer_density.step()

        if self.trainer.is_last_batch:
            (
                scheduler_encoder,
                scheduler_density,
                scheduler_offset,
                scheduler_m_scale,
            ) = self.lr_schedulers()
            scheduler_m_scale.step()
            scheduler_offset.step()
            if step < phase1_end:
                pass
            elif step < phase2_end:
                scheduler_encoder.step()
            elif step < phase3_end:
                scheduler_density.step()
            else:
                scheduler_encoder.step()
                scheduler_density.step()

        for tag, key in {
            "train/loss": "loss",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        self.log(
            "train/h_scale", self.model.h_scale.value, on_step=True, on_epoch=False
        )
        self.log(
            "train/m_scale", self.model.m_scale.value, on_step=True, on_epoch=False
        )
        self.log(
            "train/m_offset", self.model.m_offset.value, on_step=True, on_epoch=False
        )

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
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer_encoder = torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 1e-4,
                },
            ],
        )

        optimizer_density = torch.optim.AdamW(
            [
                {
                    "params": self.model.density.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 1e-4,
                },
            ],
        )

        optimizer_offset = torch.optim.AdamW(
            [
                {"params": self.model.h_scale.parameters()},
                {"params": self.model.m_offset.parameters()},
            ],
            lr=self.hparams["lr_scale"],
        )

        optimizer_m_scale = torch.optim.AdamW(
            [{"params": self.model.m_scale.parameters()}],
            lr=self.hparams["lr_scale"],
        )

        scheduler_encoder = torch.optim.lr_scheduler.StepLR(
            optimizer_encoder,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        scheduler_density = torch.optim.lr_scheduler.StepLR(
            optimizer_density,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        scheduler_offset = torch.optim.lr_scheduler.StepLR(
            optimizer_offset,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        scheduler_m_scale = torch.optim.lr_scheduler.StepLR(
            optimizer_m_scale,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        return [
            optimizer_encoder,
            optimizer_density,
            optimizer_offset,
            optimizer_m_scale,
        ], [scheduler_encoder, scheduler_density, scheduler_offset, scheduler_m_scale]
