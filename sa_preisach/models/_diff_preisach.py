from __future__ import annotations

import typing

import gpytorch.constraints
import torch
from transformertf.data import TimeSeriesSample
from transformertf.utils import (
    LrSchedulerDict,
    OptimizerDict,
    configure_optimizers,
)

from ..nn import GPyConstrainedParameter
from ..utils import (
    create_triangle_mesh,
    get_states,
    make_mesh_size_function,
)
from ._base import BaseModule


class DifferentiablePreisachModel(torch.nn.Module):
    """
    Implementation of differentiable Preisach model by Ryan Roussel (SLAC).

    Parameters
    ----------
    mesh_scale : float
        Scale of the mesh. Determines the resolution of the mesh.
    m_scale_bounds : tuple[float, float], optional
        Bounds for the scaling of the magnetization. Default is (0.0, 10.0).
    offset_bounds : tuple[float, float], optional
        Bounds for the offset. Default is (-10.0, 10.0).
    h_slope_bounds : tuple[float, float], optional
        Bounds for the slope of the magnetic field. Default is (10.0, 10.0).
    use_normalized_density : bool, optional
        If True, the density is normalized to [0, 1]. Otherwise, it is positive. Default is True.
        The computed magnetization is always scaled by the total density.
    mesh_density_function : str, optional
        Function to compute the mesh density. Options are 'constant', 'exponential', and 'default'.
        Default is 'default'.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        mesh_scale: float,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        h_slope_bounds: tuple[float, float] = (10.0, 10.0),
        use_normalized_density: bool = True,
        mesh_density_function: typing.Literal[
            "constant", "exponential", "default"
        ] = "default",
    ) -> None:
        super().__init__()
        self.mesh_scale = mesh_scale
        self.m_scale_bounds = m_scale_bounds
        self.offset_bounds = offset_bounds
        self.h_slope_bounds = h_slope_bounds

        density_function = make_mesh_size_function(mesh_density_function)

        # Create the mesh
        mesh = create_triangle_mesh(
            mesh_density_function=density_function,
            mesh_scale=self.mesh_scale,
        )
        self.mesh = torch.tensor(mesh, dtype=torch.float32)

        # Initialize the Preisach model
        density = torch.zeros(len(self.mesh), dtype=torch.float32)
        torch.nn.init.uniform(density)

        self.density = GPyConstrainedParameter(
            density,
            constraint=gpytorch.constraints.Interval(0, 1.0)
            if use_normalized_density
            else gpytorch.constraints.Positive(),
            requires_grad=True,
        )

        self.m_scale = GPyConstrainedParameter(
            torch.tensor(1.0),
            constraint=gpytorch.constraints.Interval(*self.m_scale_bounds),
            requires_grad=False,
        )

        self.offset = GPyConstrainedParameter(
            torch.tensor(0.0),
            constraint=gpytorch.constraints.Interval(*self.offset_bounds),
            requires_grad=False,
        )

        self.h_slope = GPyConstrainedParameter(
            torch.tensor(1.0),
            constraint=gpytorch.constraints.Interval(*self.h_slope_bounds),
            requires_grad=False,
        )

    @typing.overload
    def forward(
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(  # noqa: PLR0913
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if states is None:
            states = get_states(
                h=h,
                alpha=self.mesh[:, 1],
                beta=self.mesh[:, 0],
                temp=temp,
                current_state=initial_state,
                current_field=m0,
                dtype=h.dtype,
                training=self.training,
            )

        magnetization = self.predict_magnetization(states, h)

        if return_states:
            return magnetization, states

        return magnetization

    def predict_magnetization(
        self,
        states: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts the magnetization of the system based on the current states and magnetic field.

        The magnetization is computed as the weighted sum of the states, where the weights are
        given by the optimized density values. The result is then scaled by the total density and
        adjusted by the scaling factor and offset w.r.t the auxiliary magnetic field.

        Parameters
        ----------
        states : torch.Tensor
            The current states of the system. (shape: [N, M])
        h : torch.Tensor
            The magnetic field applied to the system. (shape: [N, 1])

        Returns
        -------
        torch.Tensor
            The predicted magnetization of the system. (shape: [N, 1])
        """
        m = torch.sum(self.density.value * states, dim=-1) / torch.sum(
            self.density.value
        )
        return (
            self.m_scale.value * m.reshape(h.shape)
            + self.offset.value
            + h * self.h_slope.value
        )


class DifferentiablePreisach(BaseModule):
    def __init__(  # noqa: PLR0913
        self,
        *,
        mesh_scale: float,
        temp: float = 1e-3,
        m_scale_bounds: tuple[float, float] = (0.0, 10.0),
        offset_bounds: tuple[float, float] = (-10.0, 10.0),
        h_slope_bounds: tuple[float, float] = (-10.0, 10.0),
        use_normalized_density: bool = True,
        mesh_density_function: typing.Literal[
            "constant", "exponential", "default"
        ] = "default",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        optimizer: str = "adam",
        use_step_lr: bool = False,
        lr_step_size: int = 100,
        lr_gamma: float = 0.95,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DifferentiablePreisachModel(
            mesh_scale=mesh_scale,
            m_scale_bounds=m_scale_bounds,
            offset_bounds=offset_bounds,
            h_slope_bounds=h_slope_bounds,
            use_normalized_density=use_normalized_density,
            mesh_density_function=mesh_density_function,
        )

        # internal state used for training only
        self._states: torch.Tensor | None = None

    @typing.overload
    def forward(
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(  # noqa: PLR0913
        self,
        h: torch.Tensor,
        m0: torch.Tensor | float | None = None,
        temp: float = 1e-3,
        states: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            h=h,
            m0=m0,
            temp=temp,
            states=states,
            initial_state=initial_state,
            return_states=return_states,
        )

    def on_train_epoch_start(self) -> None:
        if self.global_step > 0:
            return super().on_train_epoch_start()

        # set initial states that are needed for training
        assert self.trainer.train_dataloader is not None
        batch = next(iter(self.trainer.train_dataloader))

        x = batch["input"].squeeze(0).squeeze(-1)

        states = get_states(
            h=x,
            alpha=self.model.mesh[:, 1],
            beta=self.model.mesh[:, 0],
            temp=self.hparams["temp"],
            current_state=None,
            current_field=None,
            dtype=x.dtype,
            training=True,
        )

        # set the states to the model
        self._states = states

        return super().on_train_epoch_start()

    def common_step(
        self, batch: TimeSeriesSample, batch_idx: int, *, reuse_states: bool = False
    ) -> dict[str, torch.Tensor]:
        x = batch["input"]
        if "target" not in batch:
            msg = "Target not found in batch. This is required for training."
            raise ValueError(msg)

        y = batch["target"]

        # get rid of batch dimension
        x = x.squeeze(0).squeeze(-1)
        y = y.squeeze(0).squeeze(-1)

        y_hat = self.model(
            h=x,
            temp=self.hparams["temp"],
            states=self._states if reuse_states else None,
            initial_state=None,
            return_states=False,
        )

        loss = torch.nn.functional.mse_loss(y_hat, y)

        return {
            "loss": loss,
            "x": x,
            "y_hat": y_hat,
            "y": y,
            "density": self.model.density.value.detach().clone(),
        }

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.common_step(batch, batch_idx, reuse_states=True)

        self.log("train/loss", out["loss"], prog_bar=True, on_step=True)

        return out

    def validation_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.common_step(batch, batch_idx, reuse_states=False)

        self.log("validation/loss", out["loss"], prog_bar=True)

        return out

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> OptimizerDict | LrSchedulerDict | torch.optim.Optimizer:
        optimizer = configure_optimizers(
            self.hparams["optimizer"],
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            momentum=self.hparams["momentum"],
        )(self.parameters())

        if self.hparams["use_step_lr"]:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams["lr_step_size"],
                gamma=self.hparams["lr_gamma"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        return optimizer
