from __future__ import annotations

import logging
import typing

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from transformertf.data import TimeSeriesDataset

from ..models import (
    DifferentiablePreisach,
    DifferentiablePreisachNN,
    SelfAdaptivePreisach,
)

log = logging.getLogger(__name__)


class PlotHysteresisCallback(L.pytorch.callbacks.Callback):
    def __init__(
        self,
        *,
        validate_every_n_epochs: int = 1,
        hysteron_scatter: bool = True,
        plot_unscaled: bool = False,
    ) -> None:
        super().__init__()
        self.validate_every_n_epochs = validate_every_n_epochs
        self.hysteron_scatter = hysteron_scatter

    def on_validation_epoch_end(  # type: ignore[override]
        self,
        trainer: L.Trainer,
        pl_module: SelfAdaptivePreisach | DifferentiablePreisach,
    ) -> None:
        if not trainer.is_global_zero:
            return super().on_validation_epoch_end(trainer, pl_module)

        if trainer.current_epoch % self.validate_every_n_epochs != 0:
            return super().on_validation_epoch_end(trainer, pl_module)

        dataloader = trainer.train_dataloader
        if dataloader is None:
            if trainer.global_step > 0:
                msg = "No datamodule found"
                log.error(msg)
            return super().on_validation_epoch_end(trainer, pl_module)

        dataset = typing.cast(TimeSeriesDataset, dataloader.dataset)

        h_transform, b_transform = list(dataset.transforms.values())

        # get the last validation output
        last_output = pl_module.validation_outputs[-1]
        x = last_output["x"]
        y = last_output["y"]
        y_hat = last_output["y_hat"]
        x = x.squeeze(0).detach().cpu()
        y = y.squeeze(0).detach().cpu()
        y_hat = y_hat.squeeze(0).detach().cpu()

        x = h_transform.inverse_transform(x)
        y = b_transform.inverse_transform(x, y)
        y_hat = b_transform.inverse_transform(x, y_hat)

        fig_hysteresis = plot_hysteresis(x, y, y_hat)

        self._log_figure(trainer, fig_hysteresis, tag="validation/hysteresis")
        plt.close(fig_hysteresis)

        # plot the hysteron density
        density = last_output["density"]
        density = density.squeeze(0).detach().cpu()

        if isinstance(pl_module, SelfAdaptivePreisach):
            alpha = pl_module.model.alpha.value
            beta = pl_module.model.beta.value
        elif isinstance(pl_module, DifferentiablePreisach | DifferentiablePreisachNN):
            alpha = pl_module.model.mesh[:, 1]
            beta = pl_module.model.mesh[:, 0]
        else:
            msg = "Model not supported"
            log.error(msg)
            return super().on_validation_epoch_end(trainer, pl_module)

        alpha = alpha.detach().cpu()
        beta = beta.detach().cpu()

        fig_density = plot_hysteron_density(
            alpha,
            beta,
            density,
        )

        self._log_figure(trainer, fig_density, tag="validation/hysteron_density")
        plt.close(fig_density)

        if self.hysteron_scatter:
            # plot the hysteron scatter
            fig_scatter = plot_hysteron_scatter(
                alpha,
                beta,
            )
            self._log_figure(trainer, fig_scatter, tag="validation/hysteron_scatter")
            plt.close(fig_scatter)

        return super().on_validation_epoch_end(trainer, pl_module)

    def _log_figure(
        self, trainer: L.Trainer, fig: matplotlib.figure.Figure, tag: str
    ) -> None:
        if trainer.logger is None:
            return
        if isinstance(trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            trainer.logger.experiment[tag].append(fig)
        elif isinstance(trainer.logger, L.pytorch.loggers.TensorBoardLogger):
            trainer.logger.experiment.add_figure(
                tag, fig, global_step=trainer.global_step
            )
        else:
            msg = "Logger not supported"
            raise NotImplementedError(msg)


def plot_hysteresis(
    x: torch.Tensor,
    y: torch.Tensor,
    y_hat: torch.Tensor,
) -> matplotlib.figure.Figure:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()

    ax1.plot(x, y, label="data")
    ax1.plot(x, y_hat, label="model")

    ax1.set_xlabel("I [A]")
    ax1.set_ylabel("B [T]")

    ax1.legend()
    ax1.grid()
    ax1.set_title("Hysteresis")

    ax2.plot(x, y - y_hat, label="error")
    ax2.set_xlabel("I [A]")
    ax2.set_ylabel("B [T]")

    return fig


def plot_hysteron_density(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    density: torch.Tensor,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    density = density.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    beta = beta.detach().cpu().numpy()

    c = ax.tripcolor(
        beta,
        alpha,
        density,
        # shading="gouraud",
        cmap="viridis",
        # vmin=0.0,
        # vmax=1.0,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$\\alpha$")

    ax.set_title("Hysteron density")

    return fig


def plot_hysteron_scatter(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    alpha = alpha.detach().cpu().numpy()
    beta = beta.detach().cpu().numpy()

    ax.scatter(
        beta,
        alpha,
        s=1,
        c="black",
        alpha=0.5,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        label="$\\alpha = \\beta$",
        linewidth=0.5,
    )
    ax.plot(
        [0, 0],
        [0, 1],
        color="black",
        label="$\\beta = 0$",
        linewidth=0.5,
    )
    ax.plot(
        [0, 1],
        [1, 1],
        color="black",
        label="$\\alpha = 1$",
        linewidth=0.5,
    )

    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$\\alpha$")
    ax.set_title("Hysteron scatter")

    return fig
