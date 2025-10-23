from __future__ import annotations

import itertools
import typing

import einops
import gpytorch.constraints
import numpy as np
import torch
from transformertf.data import TimeSeriesSample
from transformertf.nn.functional import mse_loss

from ..data import PreisachDataModule
from ..nn import BinaryParameter, GPyConstrainedParameter, ResNetMLP
from ..utils import (
    create_triangle_mesh,
    get_states,
    make_mesh_size_function,
    set_requires_grad,
)
from ._base import BaseModule

CPU_DEVICE = torch.device("cpu")


class DifferentiablePreisachNNModel(torch.nn.Module):
    def __init__(  # noqa: PLR0913
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
    ) -> None:
        super().__init__()

        self.mesh = torch.from_numpy(
            create_triangle_mesh(
                mesh_size,
                mesh_density_function=mesh_density_function
                if callable(mesh_density_function)
                else make_mesh_size_function(mesh_density_function),
            )
        ).float()
        n_mesh_points = self.mesh.shape[0]

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

        self.m_scale = GPyConstrainedParameter(
            torch.tensor(1.0),
            constraint=gpytorch.constraints.Interval(*m_scale_bounds),
            requires_grad=True,
        )
        self.m_offset = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*offset_bounds),
        )

        initial_state = torch.zeros((1, n_mesh_points), dtype=torch.float32)
        torch.nn.init.xavier_uniform_(initial_state.data)
        initial_state = initial_state.squeeze(0)  # hack to use torch.nn.init

        self.initial_state = BinaryParameter(
            initial_state,
        )

    def forward(
        self,
        x: torch.Tensor,
        y0: torch.Tensor | float = 0.0,
        *,
        temp: float = 1e-3,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is not None:
            states = state
        else:
            states = get_states(
                x,
                self.mesh[:, 1],
                self.mesh[:, 0],
                temp=temp,
                current_state=self.initial_state.value,
                current_field=y0,
                dtype=torch.float32,
                training=self.training,
            )

        density = self.density(self.mesh)
        density = self.density_activation(density)
        density = einops.rearrange(density, "n 1 -> 1 n")

        m = torch.sum(density * states, dim=-1) / torch.sum(density, dim=-1)

        # return self.m_scale.value * m + self.m_offset.value + self.h_scale.value * x, density, states
        return self.m_scale.value * m + self.m_offset.value, density, states


class DifferentiablePreisachNN(BaseModule):
    model: DifferentiablePreisachNNModel

    """
    Parameters
    ----------

    mesh_scale : float
        The scale of the mesh. This is used to create the mesh for the Preisach model.
    hidden_dim : int
        The number of hidden units in the MLP used to model the Preisach density.
    num_layers : int
        The number of layers in the MLP used to model the Preisach density.
    temp : float
        The temperature parameter for the hysteron activation function (tanh).
    lr : float
        The learning rate for the main optimizer.
    lr_scale : float
        The learning rate for the scale and offset parameters.
    lr_step_interval : int
        The interval at which to step the learning rate scheduler.
    lr_gamma : float
        The factor by which to scale the learning rate at each step.
    m_scale_bounds : tuple[float, float]
        The bounds for the scale parameter of the Preisach model (scale of M).
    offset_bounds : tuple[float, float]
        The bounds for the offset parameter of the Preisach model (offset of M).
    normalized_density : bool
        Whether to normalize the density function.
    mesh_density_function : typing.Literal["constant", "default", "exponential"] | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        The function to use for the mesh density.
        If a string is provided, it must be one of "constant", "default", or "exponential".
        If a callable is provided, it must take the mesh and the scale as input and return the density.
    compile_model : bool
        Whether to compile the model using torch.compile.
    resample_every : int
        The number of epochs after which to resample the mesh.
    freeze_initial_state_after : int
        The number of epochs after which to freeze the initial state.
    update_initial_state_every : int
        When to regularly update the initial state.
    loss_weights : torch.Tensor | None
        The weights to use for the loss function. If None, weights are not used.
    _n_train_samples : int
        The number of training samples. If provided, loss_weights must be None.
        This is used to create a loss weight for each training sample, and is passed by LightningCLI.
    """

    def __init__(  # noqa: PLR0913
        self,
        mesh_scale: float,
        *,
        hidden_dim: int,
        num_layers: int = 3,
        temp: float = 1e-3,
        lr: float = 1e-2,
        lr_scale: float = 1e-3,
        lr_sa: float = 1e-2,
        lr_step_interval: int = 100,
        lr_gamma: float = 0.9,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        normalized_density: bool = True,
        mesh_density_function: typing.Literal["constant", "default", "exponential"]
        | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] = "default",
        compile_model: bool = True,
        resample_every: int = 10,
        freeze_initial_state_after: int = 100,
        update_initial_state_every: int = 100,
        loss_weights: torch.Tensor | None = None,
        n_train_samples: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_weights"])

        self.model = DifferentiablePreisachNNModel(
            mesh_size=mesh_scale,
            hidden_dim=hidden_dim,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            normalized_density=normalized_density,
            mesh_density_function=mesh_density_function,
        )

        if n_train_samples > 0:
            if loss_weights is not None:
                msg = "loss_weights must be None if _n_train_samples is provided"
                raise ValueError(msg)
            loss_weights = torch.ones(
                n_train_samples, dtype=torch.float32, device=self.device
            )
        else:
            loss_weights = loss_weights.float() if loss_weights is not None else None

        self.loss_weights = (
            torch.nn.Parameter(loss_weights, requires_grad=True)
            if loss_weights is not None
            else None
        )
        self.automatic_optimization = False

        self.states: torch.Tensor | None = None

    def on_fit_start(self) -> None:
        # if self.hparams["compile_model"]:
        #     self.model = torch.compile(self.model)

        self.model.mesh = self.model.mesh.to(self.device)

        return super().on_fit_start()

    def forward(
        self,
        x: torch.Tensor,
        y0: torch.Tensor | float = 0.0,
        *,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x, y0, temp=temp, state=states)

    def common_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        states: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        x = batch["input"][0, ..., 0]
        y = batch["target"][0, ..., 0]
        y_hat, density, states = self(
            x, y0=y[0], temp=self.hparams["temp"], states=states
        )
        loss1 = mse_loss(y_hat, y, weight=weights if weights is not None else None)
        unweighted_loss = torch.nn.functional.mse_loss(y_hat, y)

        return {
            "loss": loss1,
            "loss1": loss1,
            "y_hat": y_hat,
            "y": y,
            "x": x,
            "density": density.detach().clone(),
            "states": states.detach().clone(),
            "unweighted_loss": unweighted_loss,
        }

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.common_step(
            batch,
            batch_idx,
            states=self.states  # reuse previous states for faster training
            if self.states is not None
            and self.current_epoch > self.hparams["freeze_initial_state_after"]
            else None,
            weights=self.loss_weights if self.loss_weights is not None else None,
        )
        loss = out["loss"]

        optimizers = self.optimizers()

        if isinstance(optimizers, list):
            optimizer1, optimizer2 = optimizers
        else:
            optimizer1 = optimizers
            optimizer2 = None

        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.zero_grad()

        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(optimizer1, gradient_clip_val=1.0)
        optimizer1.step()

        if optimizer2 is not None:
            # negate gradients
            self.negate_gradients()
            optimizer2.step()

        # step lr scheduler
        schedulers = self.lr_schedulers()
        self.lr_scheduler_step(schedulers, metric=None)

        self.states = out["states"]

        for tag, key in {
            "train/loss": "loss",
            "train/mse": "unweighted_loss",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=True, on_epoch=False)

        # if loss weights are used, log the magnitude of the weights
        if self.loss_weights is not None:
            self.log(
                "train/loss_weights",
                torch.mean(self.loss_weights.data),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        return out

    def validation_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.common_step(batch, batch_idx)

        for tag, key in {
            "validation/loss": "loss",
        }.items():
            self.log(tag, out[key], prog_bar=True, on_step=False, on_epoch=True)

        return out

    def negate_gradients(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad = -param.grad

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer1 = torch.optim.AdamW(
            [
                {
                    "params": self.model.density.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 1e-4,  # only weight decay for the MLP
                },
                {
                    "params": self.model.initial_state.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": 0.0,
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
        optimizer2 = (
            torch.optim.SGD([
                {
                    "params": self.loss_weights,
                    "lr": self.hparams["lr_sa"],
                    "weight_decay": 0.0,
                }
            ])
            if self.loss_weights is not None
            else None
        )
        scheduler1 = torch.optim.lr_scheduler.StepLR(
            optimizer1,
            step_size=self.hparams["lr_step_interval"],
            gamma=self.hparams["lr_gamma"],
        )

        optimizers = [optimizer1]
        if optimizer2 is not None:
            optimizers.append(optimizer2)
        return optimizers, [scheduler1]

    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.hparams["resample_every"] == 0:
            self.model.mesh = self.resample_mesh(
                self.hparams["mesh_scale"],
                mesh_density_function=self.hparams["mesh_density_function"],
                randomize=True,
                device=self.device,
            )

        if self.current_epoch == self.hparams["freeze_initial_state_after"]:
            set_requires_grad(self.model.initial_state, flag=False)

        if (
            self.current_epoch % self.hparams["update_initial_state_every"] == 0
            and self.current_epoch > self.hparams["freeze_initial_state_after"]
        ):
            dataloader = self.trainer.train_dataloader
            if dataloader is None:
                msg = "No training dataloader found"
                raise ValueError(msg)
            batch = typing.cast(TimeSeriesSample[torch.Tensor], dataloader.dataset[0])
            assert "target" in batch

            new_initial_state, _ = DifferentiablePreisachNN.fit_initial_state(
                self,
                train_h=batch["input"][..., 0],
                train_b=batch["target"][..., 0],
                n_epochs=10,
                normalize=False,
                loss_weights=self.loss_weights.data
                if self.loss_weights is not None
                else None,
            )
            self.model.initial_state.load_state_dict(new_initial_state)

        return super().on_train_epoch_start()

    @staticmethod
    def resample_mesh(
        mesh_scale: float,
        *,
        mesh_density_function: typing.Literal["constant", "default"]
        | typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] = "default",
        randomize: float = True,
        device: torch.device = CPU_DEVICE,
    ) -> torch.Tensor:
        mesh = create_triangle_mesh(
            mesh_scale,
            make_mesh_size_function(mesh_density_function)
            if not callable(mesh_density_function)
            else mesh_density_function,
        )

        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)

        if randomize:
            mesh[:, 0] += (torch.rand(mesh.shape[0], device=device) - 0.5) * 2e-2
            mesh[:, 1] += (torch.rand(mesh.shape[0], device=device) - 0.5) * 2e-2
            mesh[:, 0] = torch.clamp(mesh[:, 0], min=0.0, max=1.0)
            mesh[:, 1] = torch.clamp(mesh[:, 1], min=0.0, max=1.0)

            # clamp values so that second column is always greater than first
            mesh[:, 1] = torch.clamp(
                mesh[:, 1], min=mesh[:, 0], max=torch.tensor(1.0, device=device).float()
            )
            mesh[:, 0] = torch.clamp(
                mesh[:, 0], min=torch.tensor(0.0, device=device).float(), max=mesh[:, 1]
            )

        return mesh

    @staticmethod
    @typing.overload
    def fit_initial_state(
        model: DifferentiablePreisachNN,
        train_h: torch.Tensor,
        train_b: torch.Tensor,
        *,
        n_epochs: int = 10,
        normalize: typing.Literal[True],
        datamodule: PreisachDataModule,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]: ...

    @staticmethod
    @typing.overload
    def fit_initial_state(
        model: DifferentiablePreisachNN,
        train_h: torch.Tensor,
        train_b: torch.Tensor,
        *,
        n_epochs: int = 10,
        normalize: typing.Literal[False] = False,
        datamodule: None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]: ...

    @staticmethod
    def fit_initial_state(  # noqa: PLR0913
        model: DifferentiablePreisachNN,
        train_h: torch.Tensor,
        train_b: torch.Tensor,
        *,
        n_epochs: int = 10,
        normalize: bool = False,
        datamodule: PreisachDataModule | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if normalize and datamodule is None:
            msg = "datamodule must be provided when normalize=True"
            raise ValueError(msg)

        was_model_state = model.training
        model.train()

        optimizer = torch.optim.Adam(model.model.parameters(), lr=model.hparams["lr"])

        if normalize:
            assert datamodule is not None
            train_h_norm = (
                datamodule.transforms[datamodule.known_covariates[0].name]
                .transform(train_h)
                .float()
            )
            train_b_norm = datamodule.target_transform.transform(
                train_h, train_b
            ).float()
        else:
            train_h_norm = train_h
            train_b_norm = train_b

        # save the initial requires grad so we can reset it later
        density_requires_grad = next(model.model.density.parameters()).requires_grad
        set_requires_grad(model.model.density, flag=False)

        # disable gradient for everything except the initial state
        s0_requires_grad = next(model.model.initial_state.parameters()).requires_grad
        set_requires_grad(model.model.initial_state, flag=True)
        set_requires_grad(model.model.m_scale, flag=False)
        set_requires_grad(model.model.m_offset, flag=False)

        initial_states = model.model.initial_state.state_dict()

        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()

            y_hat, *_ = model(
                train_h_norm,
                y0=train_b_norm[0],
                temp=model.hparams["temp"],
            )

            loss = mse_loss(y_hat, train_b_norm, weight=loss_weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # re-set the initial requires_grad
        set_requires_grad(model.model.density, flag=density_requires_grad)
        set_requires_grad(model.model.initial_state, flag=s0_requires_grad)

        new_initial_state = model.model.initial_state.state_dict()
        model.model.initial_state.load_state_dict(initial_states)

        if not was_model_state:
            model.eval()

        return new_initial_state, torch.tensor(losses, dtype=torch.float32)
