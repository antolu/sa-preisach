from __future__ import annotations

import logging
import typing

import gpytorch.constraints
import numpy as np
import torch
from transformertf.nn.functional import mse_loss

if typing.TYPE_CHECKING:
    from ..data._cycled_datamodule import CycleBatch

from ..nn import GPyConstrainedParameter, ResNetMLP
from ..priors import CompositeDensityPrior, DensityPrior, SymmetryDensityPrior
from ..utils import create_triangle_mesh, get_states, make_mesh_size_function
from ._base import BaseModule

log = logging.getLogger(__name__)

MIN_VARIANCE_THRESHOLD = 1e-6

class PreisachNNModel(torch.nn.Module):
    """
    Preisach model without encoder.

    All hysterons start fully deactivated (state = -1) at a configurable
    initial field value. This is appropriate when every training cycle begins
    from the same operating point (e.g. fully demagnetized / minimum field).

    Inputs and outputs use MinMax-normalized quantities:
      H_norm in [0, 1], B_norm in [0, 1], M in [-1, 1].
    """

    def __init__(
        self,
        *,
        mesh_size: float,
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

        self.density_prior: DensityPrior | None = None

    def _perturb_mesh(self, mesh_coords: torch.Tensor) -> torch.Tensor:
        if not self.training or np.isclose(self.mesh_perturbation_std, 0.0):
            return mesh_coords
        noise = torch.randn_like(mesh_coords) * self.mesh_perturbation_std
        perturbed = torch.clamp(mesh_coords + noise, 0.0, 1.0)
        beta = torch.minimum(perturbed[..., 0], perturbed[..., 1])
        alpha = perturbed[..., 1]
        return torch.stack([beta, alpha], dim=-1)

    def get_batched_mesh_coords(self, batch_size: int) -> torch.Tensor:
        return self.base_mesh.unsqueeze(0).expand(batch_size, -1, -1)

    def density_from_mesh(
        self, mesh_coords: torch.Tensor, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        if beta is not None:
            mesh_coords = torch.cat([beta.unsqueeze(-1), mesh_coords], dim=-1)
        density = self.density(mesh_coords)
        density = self.density_activation(density)
        return density.squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        initial_states: torch.Tensor | None = None,
        initial_field: float | torch.Tensor = 0.0,
        density_override: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h:
            H_norm sequence, shape [batch, seq_len].
        initial_states:
            Hysteron states at t=0, shape [batch, n_mesh]. When None all
            hysterons start fully deactivated (-1). Pass the terminal states
            from a previous window for multi-window rollouts.
        initial_field:
            H_norm value immediately before the sequence starts.  Used by
            get_states to determine the first sweep direction.  Scalar or
            [batch] tensor. Default 0.0 (bottom of normalized range).
        density_override:
            Substitute density for M computation (e.g. a mock density during
            early training experiments). Shape [batch, n_mesh].
        temp:
            Sigmoid temperature for soft hysteron switching.

        Returns
        -------
        (B_norm, density, M, initial_states, mesh_coords)
        """
        batch_size, seq_len = h.shape

        mesh_coords = self._perturb_mesh(self.get_batched_mesh_coords(batch_size))
        density = self.density_from_mesh(mesh_coords)

        if initial_states is None:
            initial_states = -torch.ones(
                batch_size, self.n_mesh_points, device=h.device, dtype=h.dtype
            )
        elif initial_states.shape != (batch_size, self.n_mesh_points):
            msg = (
                f"initial_states shape {tuple(initial_states.shape)} does not "
                f"match (batch_size, n_mesh_points) = ({batch_size}, {self.n_mesh_points})"
            )
            raise ValueError(msg)

        if isinstance(initial_field, torch.Tensor):
            y0 = initial_field.squeeze(-1).to(h.device)
            if y0.dim() == 0:
                y0 = y0.expand(batch_size)
        else:
            y0 = torch.full((batch_size,), initial_field, device=h.device, dtype=h.dtype)

        alpha = mesh_coords[..., 1]
        beta = mesh_coords[..., 0]

        h_cpu = h.cpu()
        alpha_cpu = alpha.cpu()
        beta_cpu = beta.cpu()
        initial_states_cpu = initial_states.cpu()
        y0_cpu = y0.cpu()

        batch_states = [
            get_states(
                h=h_cpu[b],
                alpha=alpha_cpu[b],
                beta=beta_cpu[b],
                current_state=initial_states_cpu[b],
                current_field=y0_cpu[b],
                temp=temp,
                dtype=torch.float32,
                training=self.training,
            )
            for b in range(batch_size)
        ]
        states = torch.stack(batch_states, dim=0).to(h.device)

        density_for_m = density_override if density_override is not None else density
        density_sum = density_for_m.sum(dim=-1, keepdim=True)
        m = torch.sum(density_for_m.unsqueeze(1) * states, dim=-1) / density_sum

        b_out = self.h_scale.value * h + self.m_scale.value * m + self.m_offset.value

        return b_out, density, m, initial_states, mesh_coords


class PreisachNN(BaseModule):
    model: PreisachNNModel
    supports_multiple_validation_dataloaders: bool = True

    def __init__(
        self,
        mesh_scale: float,
        *,
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
        gradient_clip_val: float = 1.0,
        initial_field: float = 0.0,
        density_prior: DensityPrior | None = None,
        fit_scale_offset: bool = False,
        temp_min: float | None = None,
        temp_anneal_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["density_prior"])

        self.model = PreisachNNModel(
            mesh_size=mesh_scale,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            normalized_density=normalized_density,
            mesh_density_function=mesh_density_function,
            mesh_perturbation_std=mesh_perturbation_std,
            fit_scale_offset=fit_scale_offset,
        )

        self.model.density_prior = density_prior
        self._prior_leaves: list[DensityPrior] = []
        self._prior_leaf_by_key: dict[str, DensityPrior] = {}
        self._prior_key_counts: dict[str, int] = {}
        if density_prior is not None:
            self._collect_prior_leaves(density_prior)

        self.automatic_optimization = False

    def _collect_prior_leaves(self, prior: DensityPrior) -> None:
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
                if k in self._prior_leaf_by_key:
                    count = self._prior_key_counts.get(k, 1)
                    if count == 1:
                        existing = self._prior_leaf_by_key.pop(k)
                        self._prior_leaf_by_key[f"{k}_0"] = existing
                    self._prior_leaf_by_key[f"{k}_{count}"] = prior
                    self._prior_key_counts[k] = count + 1
                else:
                    self._prior_leaf_by_key[k] = prior
                    self._prior_key_counts[k] = 1
            self._prior_leaves.append(prior)

    def on_fit_start(self) -> None:
        log.info("Number of mesh points: %s", self.model.n_mesh_points)
        self.logger.log_hyperparams(self.hparams)

    def forward(
        self,
        h: torch.Tensor,
        initial_states: torch.Tensor | None = None,
        initial_field: float | torch.Tensor = 0.0,
        density_override: torch.Tensor | None = None,
        *,
        temp: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(
            h=h,
            initial_states=initial_states,
            initial_field=initial_field,
            density_override=density_override,
            temp=temp,
        )

    def _current_temp(self) -> float:
        step = self.global_step if self.training else float("inf")
        temp_min = self.hparams["temp_min"]
        anneal_steps = self.hparams["temp_anneal_steps"]
        if temp_min is not None and anneal_steps > 0:
            progress = min(step / anneal_steps, 1.0)
            return self.hparams["temp"] + (temp_min - self.hparams["temp"]) * progress
        return self.hparams["temp"]

    def common_step(
        self,
        batch: CycleBatch,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        inp = batch["input"]          # [batch, seq_len, n_features]
        target = batch["target"]      # [batch, seq_len, 1]
        lengths: torch.Tensor | None = batch.get("lengths")

        h = inp[..., 0]               # [batch, seq_len]
        target_squeezed = target.squeeze(-1)

        y_hat, density, _m, initial_states, mesh_coords = self(
            h=h,
            initial_field=self.hparams["initial_field"],
            temp=self._current_temp(),
        )

        length_mask: torch.Tensor | None = None
        if lengths is not None:
            T = y_hat.shape[1]
            length_mask = (
                torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            )

        seq_loss = mse_loss(y_hat, target_squeezed, mask=length_mask)

        prior_losses_raw: dict[str, torch.Tensor] = (
            self.model.density_prior(mesh_coords, density)
            if self.model.density_prior is not None
            else {}
        )
        prior_losses = (
            {k: v * self._prior_leaf_by_key[k].weight for k, v in prior_losses_raw.items()}
            if prior_losses_raw
            else {}
        )
        prior_loss: torch.Tensor = (
            sum(prior_losses.values())  # type: ignore[arg-type]
            if prior_losses
            else torch.zeros((), device=h.device)
        )

        loss = seq_loss + prior_loss

        with torch.no_grad():
            residuals = (y_hat - target_squeezed).detach()
            mse = mse_loss(residuals, torch.zeros_like(residuals), mask=length_mask)
            rmse = mse.sqrt()
            mae = (
                residuals.abs()[length_mask].mean()
                if length_mask is not None
                else residuals.abs().mean()
            )

        return {
            "loss": loss,
            "seq_loss": seq_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "prior_losses": {k: v.detach() for k, v in prior_losses.items()},
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "y_hat": y_hat,
            "y": target_squeezed,
            "x": h,
            "density": density.detach().clone(),
            "initial_states": initial_states.detach().clone(),
            "mesh_coords": mesh_coords.detach().clone(),
        }

    def training_step(self, batch: CycleBatch, batch_idx: int) -> torch.Tensor:
        optimizers = self.optimizers()
        optimizer_density = optimizers[0]
        optimizer_scale = optimizers[1] if self.hparams["fit_scale_offset"] else None

        out = self.common_step(batch, batch_idx)
        loss = out["loss"]

        optimizer_density.zero_grad()
        if optimizer_scale is not None:
            optimizer_scale.zero_grad()

        self.manual_backward(loss)

        self.clip_gradients(
            optimizer_density,
            gradient_clip_val=self.hparams["gradient_clip_val"],
        )
        optimizer_density.step()

        if optimizer_scale is not None:
            self.clip_gradients(
                optimizer_scale,
                gradient_clip_val=self.hparams["gradient_clip_val"],
            )
            optimizer_scale.step()

        if self.trainer.is_last_batch:
            schedulers = self.lr_schedulers()
            schedulers[0].step()
            if optimizer_scale is not None:
                schedulers[1].step()

        for tag, key in {
            "train/loss": "loss",
            "train/seq_loss": "seq_loss",
            "train/prior_loss": "prior_loss",
            "train/mse": "mse",
            "train/rmse": "rmse",
            "train/mae": "mae",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        for k, v in out["prior_losses"].items():
            self.log(f"train/prior/{k}", v, on_step=True, on_epoch=False)

        self.log("train/m_scale", self.model.m_scale.value, on_step=True, on_epoch=False)
        self.log("train/m_offset", self.model.m_offset.value, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self,
        batch: CycleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del dataloader_idx
        with torch.no_grad():
            out = self.common_step(batch, batch_idx)

        for tag, key in {
            "validation/loss": "loss",
            "validation/mse": "mse",
            "validation/rmse": "rmse",
            "validation/mae": "mae",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=False, on_epoch=True)

        return out

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer_density = torch.optim.AdamW(
            self.model.density.parameters(),
            lr=self.hparams["lr"],
            weight_decay=1e-4,
        )
        scheduler_density = torch.optim.lr_scheduler.StepLR(
            optimizer_density,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )
        optimizers: list[torch.optim.Optimizer] = [optimizer_density]
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [scheduler_density]

        if self.hparams["fit_scale_offset"]:
            scale_params = [
                self.model.h_scale.raw_parameter,
                self.model.m_scale.raw_parameter,
                self.model.m_offset.raw_parameter,
            ]
            optimizer_scale = torch.optim.AdamW(
                scale_params,
                lr=self.hparams["lr_scale"],
                weight_decay=0.0,
            )
            scheduler_scale = torch.optim.lr_scheduler.StepLR(
                optimizer_scale,
                step_size=self.hparams["lr_step_interval"],
                gamma=self.hparams["lr_gamma"],
            )
            optimizers.append(optimizer_scale)
            schedulers.append(scheduler_scale)

        return optimizers, schedulers
