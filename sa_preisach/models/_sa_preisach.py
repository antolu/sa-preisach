from __future__ import annotations

import einops
import torch
from transformertf.data import TimeSeriesSample
from transformertf.nn import MLP

from ..nn import BinaryParameter, ConstrainedParameter
from ..utils import (
    constant_mesh_size,
    create_triangle_mesh,
    get_states,
)
from ._base import BaseModule


class SelfAdaptivePreisachModel(torch.nn.Module):
    def __init__(
        self,
        mesh_size: float,
        hidden_dim: int | tuple[int, ...] = (16, 16, 16),
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        h_slope_bounds: tuple[float, float] = (0.0, 10.0),
    ) -> None:
        super().__init__()

        mesh = create_triangle_mesh(
            mesh_size,
            constant_mesh_size,
        )

        self.alpha = ConstrainedParameter(
            torch.tensor(mesh[:, 1], requires_grad=True).float(),
        )
        self.beta = ConstrainedParameter(
            torch.tensor(mesh[:, 0], requires_grad=True).float(),
        )
        n_mesh_points = mesh.shape[0]

        self.density = torch.compile(
            MLP(
                input_dim=2,
                output_dim=1,
                hidden_dim=hidden_dim,
                activation="relu",
                dropout=0.0,
            )
        )

        self.m_scale = ConstrainedParameter(
            torch.tensor(1.0),
            min_=m_scale_bounds[0],
            max_=m_scale_bounds[1],
        )
        self.m_offset = ConstrainedParameter(
            torch.tensor(1.0),
            min_=offset_bounds[0],
            max_=offset_bounds[1],
        )
        self.h_scale = ConstrainedParameter(
            torch.tensor(1.0),
            min_=h_slope_bounds[0],
            max_=h_slope_bounds[1],
        )

        initial_state = torch.zeros((1, n_mesh_points), dtype=torch.float32)
        torch.nn.init.xavier_uniform_(initial_state.data)
        initial_state = initial_state.squeeze(0)

        self.initial_state = BinaryParameter(
            initial_state,
        )

    def forward(
        self, x: torch.Tensor, y0: torch.Tensor | float = 0.0, *, temp: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = get_states(
            x,
            self.alpha.value,
            self.beta.value,
            temp=temp,
            current_state=self.initial_state.value,
            current_field=y0,
            dtype=torch.float32,
            training=self.training,
        )

        density = self.density(self.mesh)
        density = torch.nn.functional.sigmoid(density)
        density = einops.rearrange(density, "n 1 -> 1 n")

        m = torch.sum(density * states, dim=-1) / torch.sum(density, dim=-1)

        return (
            self.m_scale.value * m + self.m_offset.value + self.h_scale.value * x,
            density,
            states,
        )

    @property
    def mesh(self) -> torch.Tensor:
        return torch.cat(
            [
                self.alpha.value.unsqueeze(-1),
                self.beta.value.unsqueeze(-1),
            ],
            dim=1,
        )


class SelfAdaptivePreisach(BaseModule):
    def __init__(  # noqa: PLR0913
        self,
        mesh_size: float,
        *,
        hidden_dim: int | tuple[int, ...] = (16, 16, 16),
        temp: float = 1e-3,
        lr: float = 1e-2,
        lr_scale: float = 1e-3,
        lr_sa: float = 1e-2,
        gradient_clip: float = 1.0,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        h_slope_bounds: tuple[float, float] = (0.0, 10.0),
        dist_threshold: float = 1e-3,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SelfAdaptivePreisachModel(
            mesh_size=mesh_size,
            hidden_dim=hidden_dim,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            h_slope_bounds=h_slope_bounds,
        )

        self.loss_weights = torch.nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            requires_grad=True,
        )

        self.automatic_optimization = False

    def forward(
        self, x: torch.Tensor, y0: torch.Tensor | float = 0.0, *, temp: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x, y0, temp=temp)

    def common_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        x = batch["input"]
        y = batch["target"]
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_hat, density, states = self(x, y0=y[0], temp=self.hparams["temp"])
        loss1 = torch.nn.functional.mse_loss(y_hat, y)
        loss2 = self.parameter_constraint_loss()
        loss3 = overlap_penalty_loss(
            self.model.mesh, min_dist=1e-6, constant=self.hparams["dist_threshold"] ** 2
        )
        loss4 = torch.abs(y_hat[0] - y[0])

        loss = sum(
            self.loss_weights[i] * loss
            for i, loss in enumerate([loss1, loss2, loss3, loss4])
        )

        return {
            "loss": loss,  # type: ignore[dict-item]
            "loss_weights": self.loss_weights.detach().clone(),
            "loss1": loss1,
            "loss2": loss2,
            "loss3": loss3,
            "loss4": loss4,
            "y_hat": y_hat,
            "y": y,
            "x": x,
            "density": density.detach().clone(),
            "states": states.detach().clone(),
        }

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.common_step(batch, batch_idx)
        loss = out["loss"]

        optimizer1, optimizer2, optimizer3 = self.optimizers()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        self.manual_backward(loss)

        self.clip_gradients(
            optimizer1,
            gradient_clip_val=self.hparams["gradient_clip"],
        )
        self.clip_gradients(
            optimizer2,
            gradient_clip_val=self.hparams["gradient_clip"],
        )

        optimizer1.step()
        optimizer2.step()

        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if isinstance(lr_schedulers, list):
                for scheduler in lr_schedulers:
                    self.lr_scheduler_step(scheduler, metric=None)
            else:
                self.lr_scheduler_step(lr_schedulers, metric=None)

        self.negate_gradients()
        optimizer3.step()

        for tag, key in {
            "train/loss": "loss",
            "train/loss_mse": "loss1",
            "train/loss_alpha_geq_beta": "loss2",
            "train/loss_overlap": "loss3",
            "train/loss_i0": "loss4",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        for tag, i in {
            "train/loss_weight_mse": 0,
            "train/loss_weight_alpha_geq_beta": 1,
            "train/loss_weight_overlap": 2,
            "train/loss_weight_i0": 3,
        }.items():
            self.log(
                tag, self.loss_weights[i], prog_bar=True, on_step=True, on_epoch=False
            )

        return out

    def validation_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.common_step(batch, batch_idx)

        for tag, key in {
            "validation/loss": "loss",
            "validation/loss_mse": "loss1",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=False, on_epoch=True)

        return out

    def parameter_constraint_loss(self) -> torch.Tensor:
        return torch.sum(
            torch.nn.functional.relu(self.model.beta.value - self.model.alpha.value)
        )

    def negate_gradients(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad = -param.grad

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer1 = torch.optim.AdamW(
            [
                param
                for name, param in self.named_parameters()
                if name not in {"m_scale", "m_offset", "h_scale", "loss_weights"}
            ],
            lr=self.hparams["lr"],
            weight_decay=1e-4,
        )
        optimizer2 = torch.optim.AdamW(
            [  # type: ignore[arg-type]
                self.model.m_scale.data,
                self.model.m_offset.data,
                self.model.h_scale.data,
            ],
            lr=self.hparams["lr_scale"],
            weight_decay=1e-4,
        )
        optimizer3 = torch.optim.SGD([self.loss_weights], lr=self.hparams["lr_sa"])
        scheduler1 = torch.optim.lr_scheduler.StepLR(
            optimizer1, step_size=200, gamma=0.9
        )
        scheduler2 = torch.optim.lr_scheduler.StepLR(
            optimizer2, step_size=200, gamma=0.9
        )
        return [optimizer1, optimizer2, optimizer3], [scheduler1, scheduler2]


def overlap_penalty_loss(
    pairs: torch.Tensor, min_dist: float = 1e-6, constant: float = 1e-6
) -> torch.Tensor:
    """
    Penalize overlapping (x, y) pairs in a tensor of shape (N, 2) using 1/distance^2.

    Overlapping should only be a problem when distance is < 1e-3
    Args:
        pairs: Tensor of shape (N, 2)
        min_dist: Minimum allowed distance between pairs
        constant: Multiplication constant for the penalty
    Returns:
        penalty: Scalar tensor with the overlap penalty
    """
    # Compute pairwise distances (N, N)
    diff = pairs.unsqueeze(1) - pairs.unsqueeze(0)  # (N, N, 2)
    dists = torch.norm(diff, dim=-1)  # (N, N)

    # Ignore self-distances by setting diagonal to a large value
    dists += torch.eye(pairs.shape[0], device=pairs.device) * 1e6

    # Penalize distances below min_dist using 1/distance^2
    return torch.mean(constant / torch.pow(torch.clamp(dists, min=min_dist), 2))
