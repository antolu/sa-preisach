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
    ResNetMLP,
)
from ..priors import DensityPrior
from ..utils import (
    create_triangle_mesh,
    get_states,
    make_mesh_size_function,
)
from ._base import BaseModule

log = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")
MIN_VARIANCE_THRESHOLD = 1e-6


def phase1_loss(  # noqa: PLR0913
    initial_states: torch.Tensor,
    y0: torch.Tensor,
    mesh_coords: torch.Tensor,
    density: torch.Tensor,
    m_target: torch.Tensor,
    saturation_reg: torch.Tensor,
    prior_loss: torch.Tensor,
    *,
    aux_loss_weight: float,
    saturation_reg_weight: float,
) -> dict[str, torch.Tensor]:
    # For any H_last two regions of the Preisach plane (α ≥ β convention) are
    # unambiguous regardless of magnetic history:
    #   β < H_last  → deactivation threshold is below current field → hysteron ON  (+1)
    #   α > H_last  → activation threshold is above current field   → hysteron OFF (-1)
    # Hysterons where β ≥ H_last and α ≤ H_last are history-dependent → masked out.
    h_last = y0.unsqueeze(-1)  # [batch, 1]  (H_norm at end of context)
    alpha = mesh_coords[..., 1]  # [batch, n_mesh]  (upper / switch-up threshold)
    beta = mesh_coords[..., 0]  # [batch, n_mesh]  (lower / switch-down threshold)

    mask_pos = beta < h_last  # unambiguously ON
    mask_neg = alpha > h_last  # unambiguously OFF

    physics_target = torch.where(
        mask_pos,
        torch.ones_like(beta),
        torch.where(
            mask_neg,
            -torch.ones_like(alpha),
            torch.zeros_like(beta),  # history-dependent, masked out below
        ),
    )

    # Stratified loss: average the +1-region loss and the -1-region loss with equal
    # weight, regardless of how many hysterons fall in each region.  Without this,
    # a high H_last makes the +1 region much larger, giving a dominant +1 gradient
    # that drives the encoder toward full saturation.
    loss_pos = (
        mse_loss(initial_states[mask_pos], physics_target[mask_pos])
        if mask_pos.any()
        else torch.tensor(0.0, device=initial_states.device)
    )
    loss_neg = (
        mse_loss(initial_states[mask_neg], physics_target[mask_neg])
        if mask_neg.any()
        else torch.tensor(0.0, device=initial_states.device)
    )
    physics_loss = 0.5 * (loss_pos + loss_neg)

    density_sum = density.sum(dim=-1)
    m_initial = (density * initial_states).sum(dim=-1) / density_sum
    aux_loss = mse_loss(m_initial, m_target)

    loss = (
        physics_loss
        + aux_loss_weight * aux_loss
        + saturation_reg_weight * saturation_reg
        + prior_loss
    )
    return {"loss": loss, "aux_loss": aux_loss, "physics_loss": physics_loss}


def phase2_loss(  # noqa: PLR0913
    y_hat: torch.Tensor,
    target_squeezed: torch.Tensor,
    density: torch.Tensor,
    initial_states: torch.Tensor,
    m_target: torch.Tensor,
    saturation_reg: torch.Tensor,
    prior_loss: torch.Tensor,
    *,
    aux_loss_weight: float,
    saturation_reg_weight: float,
) -> dict[str, torch.Tensor]:
    # Phase 2+: density is being trained, switch to density-weighted mean-field
    #   M_encoder = Σ μ(α,β)·s₀(α,β) / Σ μ(α,β)
    density_sum = density.sum(dim=-1)
    m_initial = (density * initial_states).sum(dim=-1) / density_sum
    aux_loss = mse_loss(m_initial, m_target)
    physics_loss = torch.zeros(1, device=initial_states.device)

    seq_loss = mse_loss(y_hat, target_squeezed)
    loss = (
        seq_loss
        + aux_loss_weight * aux_loss
        + saturation_reg_weight * saturation_reg
        + prior_loss
    )
    return {"loss": loss, "aux_loss": aux_loss, "physics_loss": physics_loss}


def optimizer_step(  # noqa: PLR0913
    step: int,
    phase1_end: int,
    phase2_end: int,
    optimizer_encoder: torch.optim.Optimizer,
    optimizer_density: torch.optim.Optimizer,
    optimizer_scale: torch.optim.Optimizer | None,
    clip_fn: typing.Callable[[torch.optim.Optimizer], None],
) -> None:
    if step < phase1_end:
        clip_fn(optimizer_encoder)
        optimizer_encoder.step()
    elif step < phase2_end:
        clip_fn(optimizer_encoder)
        clip_fn(optimizer_density)
        optimizer_encoder.step()
        optimizer_density.step()
        if optimizer_scale is not None:
            clip_fn(optimizer_scale)
            optimizer_scale.step()
    else:
        step_offset = step - phase2_end
        if step_offset % 2 == 0:
            clip_fn(optimizer_encoder)
            optimizer_encoder.step()
        else:
            clip_fn(optimizer_density)
            optimizer_density.step()
            if optimizer_scale is not None:
                clip_fn(optimizer_scale)
                optimizer_scale.step()


class EncoderDecoderPreisachNNModel(torch.nn.Module):
    """
    Neural Preisach model with encoder-decoder architecture.

    All inputs and outputs operate on MinMax-normalized quantities:
      - H_norm ∈ [0, 1]: applied field, mapped from physical H via MinMaxScaler
      - B_norm ∈ [0, 1]: flux density, mapped from physical B via MinMaxScaler
      - M ∈ [-1, 1]: Preisach magnetization, M = Σ μ(α,β)·s(α,β) / Σ μ(α,β)
        where s ∈ {-1, +1} are hysteron states and μ > 0 is the density

    The constitutive relation (with h_scale=0 for soft magnets where μ₀H ≪ B):
      B_norm = m_scale · M + m_offset
    With m_scale=0.5, m_offset=0.5 this maps M ∈ [-1,1] → B_norm ∈ [0,1].
    """

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
        fit_scale_offset: bool = False,
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

        with torch.no_grad():
            beta_m = base_mesh[:, 0]
            alpha_m = base_mesh[:, 1]
            mock_density = torch.exp(-(alpha_m - beta_m) / 0.1)
            mock_density = mock_density / mock_density.sum()
        self.register_buffer("mock_density", mock_density)

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

        # Constitutive relation: B_norm = h_scale · H_norm + m_scale · M + m_offset
        # For soft magnetic materials (ARMCO iron), μ₀H ≪ B so h_scale ≈ 0.
        # With M ∈ [-1,1], m_scale=0.5 and m_offset=0.5 give:
        #   M = -1 (neg. saturation) → B_norm = 0
        #   M = +1 (pos. saturation) → B_norm = 1
        # fit_scale_offset=False keeps these frozen at analytical values until
        # the encoder is stable enough to not fight the scale fitting.
        self.h_scale = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=fit_scale_offset,
        )
        self.m_scale = GPyConstrainedParameter(
            torch.tensor(0.5),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=fit_scale_offset,
        )
        self.m_offset = GPyConstrainedParameter(
            torch.tensor(0.5),
            constraint=gpytorch.constraints.Interval(*offset_bounds),
            requires_grad=fit_scale_offset,
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
        if not training or np.isclose(self.mesh_perturbation_std, 0.0):
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

    def forward(  # noqa: PLR0913
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        density_override: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder-decoder Preisach model.

        Parameters
        ----------
        encoder_input : torch.Tensor
            Historical [H_norm, B_norm] sequences,
            shape [batch_size, ctxt_seq_len, 2].
            Feature 0 = H_norm (normalized applied field),
            Feature 1 = B_norm (normalized flux density).
        decoder_input : torch.Tensor
            Future H_norm sequence, shape [batch_size, tgt_seq_len, 1]
        encoder_mask : torch.Tensor, optional
            Padding mask for encoder, shape [batch_size, ctxt_seq_len]
        y0 : torch.Tensor, optional
            Last H_norm value before decoder sequence, shape [batch_size].
            Used to determine initial sweep direction in the Preisach plane.
            If None, extracted from encoder_input[:, -1, 0].
        initial_states : torch.Tensor, optional
            Override for encoder-produced initial hysteron states,
            shape [batch_size, n_mesh_points]. When provided the encoder
            is bypassed and mesh perturbation is disabled to keep states
            aligned with the mesh. Intended for multi-window rollouts
            where the terminal state of the previous window is fed in.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - B_norm predictions [batch_size, tgt_seq_len]
            - density μ(α,β) [batch_size, n_mesh_points]
            - M (unscaled magnetization) [batch_size, tgt_seq_len]
            - initial hysteron states s₀ [batch_size, n_mesh_points]
            - mesh coordinates (β,α) [batch_size, n_mesh_points, 2]
        """
        batch_size = encoder_input.shape[0]

        # Expand mesh coordinates for batch
        mesh_coords = self.get_batched_mesh_coords(
            batch_size
        )  # [batch_size, n_mesh_points, 2]

        # Perturb mesh during training for density network regularization.
        # Skip perturbation when initial_states are provided externally: a
        # caller-supplied state vector is indexed against self.base_mesh, so
        # perturbing here would desync state indices from hysteron coordinates.
        training_and_perturb = self.training and initial_states is None
        mesh_coords = self._perturb_mesh(mesh_coords, training_and_perturb)

        # Compute density weights first
        density = self.density_from_mesh(
            mesh_coords, beta=None
        )  # [batch_size, n_mesh_points]

        if initial_states is None:
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
        elif initial_states.shape != (batch_size, self.n_mesh_points):
            msg = (
                f"initial_states shape {tuple(initial_states.shape)} does not "
                f"match (batch_size, n_mesh_points) = ({batch_size}, {self.n_mesh_points})"
            )
            raise ValueError(msg)

        # y0 = last H_norm from encoder context, used to determine sweep direction
        # in get_states: if decoder h[0] > y0, sweep up (activate hysterons),
        # if h[0] < y0, sweep left (deactivate hysterons)
        y0 = encoder_input[:, -1, 0] if y0 is None else y0.squeeze(-1)  # [batch_size]

        # Ensure y0 always has batch dimension for vmap compatibility
        if y0.dim() == 0:  # scalar case when batch_size=1
            y0 = y0.unsqueeze(0)  # [1]

        # H_norm values for the decoder (the "applied field" driving the Preisach model)
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

        # density_override substitutes the learned density for M computation,
        # e.g. mock_density in phase 1 before the density network is trained.
        density_for_m = density_override if density_override is not None else density

        # Normalize density once (constant across time)
        density_sum = density_for_m.sum(dim=-1, keepdim=True)  # [batch_size, 1]

        # M = Σ μ(α,β)·s(α,β) / Σ μ(α,β), where s ∈ [-1,+1] per hysteron
        m = (
            torch.sum(density_for_m.unsqueeze(1) * states, dim=-1) / density_sum
        )  # [batch_size, tgt_seq_len]

        # B_norm = h_scale · H_norm + m_scale · M + m_offset
        b_out = self.h_scale.value * h + self.m_scale.value * m + self.m_offset.value

        return (
            b_out,
            density,
            m,
            initial_states,
            mesh_coords,
        )  # (B_norm, μ, M, s₀, mesh)

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
    supports_multiple_validation_dataloaders: bool = True

    def __init__(  # noqa: PLR0913
        self,
        mesh_scale: float,
        *,
        encoder: PreisachEncoder,
        hidden_dim: int,
        num_layers: int = 3,
        temp: float = 1e-3,
        lr: float = 1e-2,
        lr_encoder: float = 1e-3,
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
        aux_loss_weight: float = 1.0,
        saturation_reg_weight: float = 0.1,
        density_prior: DensityPrior | None = None,
        n_train_samples: int = 0,
        fit_scale_offset: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "density_prior"])

        self.model = EncoderDecoderPreisachNNModel(
            mesh_size=mesh_scale,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            normalized_density=normalized_density,
            mesh_density_function=mesh_density_function,
            mesh_perturbation_std=mesh_perturbation_std,
            fit_scale_offset=fit_scale_offset,
        )

        # Register prior as submodule so it moves to the correct device with the model.
        # SymmetryDensityPrior needs the density network injected after model construction.
        self.model.density_prior = density_prior
        self._prior_leaves: list[DensityPrior] = []
        self._prior_leaf_by_key: dict[str, DensityPrior] = {}
        if density_prior is not None:
            self._collect_prior_leaves(density_prior)

        self.automatic_optimization = False

    def _collect_prior_leaves(self, prior: DensityPrior) -> None:
        from ..priors import CompositeDensityPrior, SymmetryDensityPrior

        if isinstance(prior, CompositeDensityPrior):
            for p in prior.priors:
                self._collect_prior_leaves(p)
        else:
            if isinstance(prior, SymmetryDensityPrior):
                prior.density_net = self.model.density_from_mesh
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                dummy_mesh = self.model.base_mesh.unsqueeze(0)[:, :1, :]
                dummy_density = torch.ones(1, 1)
                try:
                    sample = prior(dummy_mesh, dummy_density)
                except Exception:
                    sample = {}
            if was_training:
                self.model.train()
            for k in sample:
                self._prior_leaf_by_key[k] = prior
            self._prior_leaves.append(prior)

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

    def forward(  # noqa: PLR0913
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        density_override: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(
            encoder_input=encoder_input,
            decoder_input=decoder_input[..., 0:1],
            encoder_mask=encoder_mask,
            y0=y0,
            initial_states=initial_states,
            density_override=density_override,
            temp=temp,
        )

    def common_step(
        self,
        batch: EncoderDecoderTargetSample[torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch

        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        target = batch["target"]
        batch_size = encoder_input.shape[0]

        encoder_lengths = batch.get("encoder_lengths")
        if encoder_lengths is not None:
            encoder_lengths = encoder_lengths[..., 0]

            encoder_mask = create_mask(
                size=encoder_input.shape[1],
                lengths=encoder_lengths,
                alignment="left",
                inverse=True,
            )
        else:
            encoder_mask = batch.get("encoder_mask")
            if encoder_mask is not None:
                encoder_mask = encoder_mask.bool()

        # encoder_input features: [H_norm, B_norm]
        # y0 = last H_norm: determines Preisach sweep direction for first decoder step
        # b_last = last B_norm: used to derive M_target for the auxiliary loss
        if encoder_lengths is not None:
            batch_indices = torch.arange(batch_size, device=encoder_input.device)
            last_indices = (encoder_lengths - 1).long()
            y0 = encoder_input[batch_indices, last_indices, 0]
            b_last = encoder_input[batch_indices, last_indices, -1]
        else:
            y0 = encoder_input[:, -1, 0]
            b_last = encoder_input[:, -1, -1]

        step = self.global_step if self.training else float("inf")
        phase1_end = self.hparams["encoder_fit_steps"]

        density_override = (
            self.model.mock_density.unsqueeze(0).expand(batch_size, -1)
            if step < phase1_end
            else None
        )

        y_hat, density, _m_unscaled, initial_states, mesh_coords = self(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            encoder_mask=encoder_mask,
            y0=y0,
            density_override=density_override,
            temp=self.hparams["temp"],
        )

        target_squeezed = target.squeeze(-1)

        # Invert B_norm = m_scale·M + m_offset → M_target = (B_norm - m_offset) / m_scale
        # Must use current parameter values so this stays correct when fit_scale_offset=True
        # and the scale/offset drift from their initial 0.5/0.5 values.
        m_target = (b_last - self.model.m_offset.value) / self.model.m_scale.value

        # Saturation regularizer: penalise states near ±1 in all phases.
        # Without this the encoder can collapse to all-+1 or all-−1, which is a
        # degenerate local minimum that satisfies most aggregated losses trivially.
        # mean(s²) pushes states toward 0 (the interior of the Preisach plane) rather
        # than the saturated boundary; weight is small so it doesn't fight the physics loss.
        saturation_reg = (initial_states**2).mean()

        prior_losses_raw: dict[str, torch.Tensor] = (
            self.model.density_prior(mesh_coords, density)
            if self.model.density_prior is not None
            else {}
        )
        prior_losses: dict[str, torch.Tensor] = (
            {
                k: v * self._prior_leaf_by_key[k].weight
                for k, v in prior_losses_raw.items()
            }
            if prior_losses_raw
            else {}
        )
        prior_loss = (
            sum(prior_losses.values())  # type: ignore[arg-type]
            if prior_losses
            else torch.zeros(1, device=density.device)
        )

        # During phase 1 the density network is not yet trained; expose mock_density
        # in the output dict so callbacks see a meaningful density instead of noise.
        density_out = (
            self.model.mock_density.unsqueeze(0).expand(batch_size, -1)
            if step < phase1_end
            else density
        )

        if step < phase1_end:
            out = phase1_loss(
                initial_states=initial_states,
                y0=y0,
                mesh_coords=mesh_coords,
                density=self.model.mock_density,
                m_target=m_target,
                saturation_reg=saturation_reg,
                prior_loss=prior_loss,
                aux_loss_weight=self.hparams["aux_loss_weight"],
                saturation_reg_weight=self.hparams["saturation_reg_weight"],
            )
        else:
            out = phase2_loss(
                y_hat=y_hat,
                target_squeezed=target_squeezed,
                density=density,
                initial_states=initial_states,
                m_target=m_target,
                saturation_reg=saturation_reg,
                prior_loss=prior_loss,
                aux_loss_weight=self.hparams["aux_loss_weight"],
                saturation_reg_weight=self.hparams["saturation_reg_weight"],
            )
        loss, aux_loss, physics_loss = out["loss"], out["aux_loss"], out["physics_loss"]

        with torch.no_grad():
            residuals = y_hat.detach() - target_squeezed.detach()
            mse = (residuals**2).mean()
            rmse = mse.sqrt()
            mae = residuals.abs().mean()

        return {
            "loss": loss,
            "aux_loss": aux_loss.detach(),
            "physics_loss": physics_loss.detach(),
            "saturation_reg": saturation_reg.detach(),
            "prior_loss": prior_loss.detach(),
            "prior_losses": {k: v.detach() for k, v in prior_losses.items()},
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "y_hat": y_hat,
            "y": target_squeezed,
            "x": decoder_input.squeeze(-1),
            "density": density_out.detach().clone(),
            "initial_states": initial_states.detach().clone(),
            "mesh_coords": mesh_coords.detach().clone(),
        }

    def training_step(
        self, batch: EncoderDecoderTargetSample[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        optimizers = self.optimizers()
        optimizer_encoder, optimizer_density = optimizers[0], optimizers[1]
        optimizer_scale = optimizers[2] if self.hparams["fit_scale_offset"] else None

        out = self.common_step(batch, batch_idx)
        loss = out["loss"]

        step = self.global_step
        phase1_end = self.hparams["encoder_fit_steps"]
        phase2_end = phase1_end + self.hparams["density_fit_steps"]

        optimizer_encoder.zero_grad()
        optimizer_density.zero_grad()
        if optimizer_scale is not None:
            optimizer_scale.zero_grad()

        self.manual_backward(loss)

        clip_fn = lambda opt: self.clip_gradients(  # noqa: E731
            opt, gradient_clip_val=self.hparams["gradient_clip_val"]
        )
        optimizer_step(
            step=step,
            phase1_end=phase1_end,
            phase2_end=phase2_end,
            optimizer_encoder=optimizer_encoder,
            optimizer_density=optimizer_density,
            optimizer_scale=optimizer_scale,
            clip_fn=clip_fn,
        )

        if self.trainer.is_last_batch:
            schedulers = self.lr_schedulers()
            schedulers[0].step()  # encoder
            if step >= phase1_end:
                schedulers[1].step()  # density
                if self.hparams["fit_scale_offset"]:
                    schedulers[2].step()  # scale/offset

        for tag, key in {
            "train/loss": "loss",
            "train/aux_loss": "aux_loss",
            "train/physics_loss": "physics_loss",
            "train/saturation_reg": "saturation_reg",
            "train/prior_loss": "prior_loss",
            "train/mse": "mse",
            "train/rmse": "rmse",
            "train/mae": "mae",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        for k, v in out["prior_losses"].items():
            self.log(f"train/prior/{k}", v, on_step=True, on_epoch=False)

        self.log(
            "train/m_scale", self.model.m_scale.value, on_step=True, on_epoch=False
        )
        self.log(
            "train/m_offset", self.model.m_offset.value, on_step=True, on_epoch=False
        )

        return loss

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del dataloader_idx
        with torch.no_grad():
            out = self.common_step(batch, batch_idx)

        for tag, key in {
            "validation/loss": "loss",
            "validation/aux_loss": "aux_loss",
            "validation/mse": "mse",
            "validation/rmse": "rmse",
            "validation/mae": "mae",
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
                    "lr": self.hparams["lr_encoder"],
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

        optimizers = [optimizer_encoder, optimizer_density]
        schedulers = [scheduler_encoder, scheduler_density]

        if self.hparams["fit_scale_offset"]:
            # Separate optimizer for h_scale, m_scale, m_offset so they can
            # be clipped and stepped independently from encoder/density.
            scale_params = [
                self.model.h_scale.raw_parameter,
                self.model.m_scale.raw_parameter,
                self.model.m_offset.raw_parameter,
            ]
            optimizer_scale = torch.optim.AdamW(
                scale_params, lr=self.hparams["lr_scale"], weight_decay=0.0
            )
            scheduler_scale = torch.optim.lr_scheduler.StepLR(
                optimizer_scale,
                step_size=self.hparams["lr_step_interval"],
                gamma=self.hparams["lr_gamma"],
            )
            optimizers.append(optimizer_scale)
            schedulers.append(scheduler_scale)

        return optimizers, schedulers
