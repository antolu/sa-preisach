from __future__ import annotations

import logging
import typing

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from transformertf.data import TimeSeriesDataset
from transformertf.data.dataset import EncoderDecoderDataset

from ..models import (
    DifferentiablePreisach,
    DifferentiablePreisachNN,
    EncoderDecoderPreisachNN,
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
        plot_training: bool = False,
        train_plot_interval: int = 100,
    ) -> None:
        super().__init__()
        self.validate_every_n_epochs = validate_every_n_epochs
        self.hysteron_scatter = hysteron_scatter
        self.plot_training = plot_training
        self.train_plot_interval = train_plot_interval

    def on_validation_epoch_end(  # type: ignore[override]
        self,
        trainer: L.Trainer,
        pl_module: SelfAdaptivePreisach
        | DifferentiablePreisach
        | DifferentiablePreisachNN
        | EncoderDecoderPreisachNN,
    ) -> None:
        if not trainer.is_global_zero:
            return super().on_validation_epoch_end(trainer, pl_module)

        if trainer.current_epoch % self.validate_every_n_epochs != 0:
            return super().on_validation_epoch_end(trainer, pl_module)

        dataloaders = trainer.val_dataloaders
        if dataloaders is None:
            if trainer.global_step > 0:
                msg = "No datamodule found"
                log.error(msg)
            return super().on_validation_epoch_end(trainer, pl_module)

        if isinstance(dataloaders, list):
            for dataloader_idx, dataloader in enumerate(dataloaders):
                outputs = pl_module.validation_outputs_by_dataloader.get(dataloader_idx)
                if not outputs:
                    continue
                self._log_validation_output(
                    trainer=trainer,
                    pl_module=pl_module,
                    dataset=typing.cast(
                        TimeSeriesDataset | EncoderDecoderDataset, dataloader.dataset
                    ),
                    output=outputs[0],
                    tag_prefix=f"validation/{dataloader_idx}",
                )
        else:
            if not pl_module.validation_outputs:
                return super().on_validation_epoch_end(trainer, pl_module)
            self._log_validation_output(
                trainer=trainer,
                pl_module=pl_module,
                dataset=typing.cast(
                    TimeSeriesDataset | EncoderDecoderDataset, dataloaders.dataset
                ),
                output=pl_module.validation_outputs[0],
                tag_prefix="validation",
            )

        return super().on_validation_epoch_end(trainer, pl_module)

    def _log_validation_output(
        self,
        *,
        trainer: L.Trainer,
        pl_module: SelfAdaptivePreisach
        | DifferentiablePreisach
        | DifferentiablePreisachNN
        | EncoderDecoderPreisachNN,
        dataset: TimeSeriesDataset | EncoderDecoderDataset,
        output: dict[str, torch.Tensor],
        tag_prefix: str,
    ) -> None:
        h_transform, b_transform = list(dataset.transforms.values())

        x_sample = output["x"][0]
        y_sample = output["y"][0]
        y_hat_sample = output["y_hat"][0]

        if x_sample.ndim > 1 and x_sample.shape[-1] > 1:
            x_sample = x_sample[..., 0]

        x_sample = x_sample.detach().cpu()
        y_sample = y_sample.detach().cpu()
        y_hat_sample = y_hat_sample.detach().cpu()

        x_inv = h_transform.inverse_transform(x_sample)
        y_inv = b_transform.inverse_transform(x_sample, y_sample)
        y_hat_inv = b_transform.inverse_transform(x_sample, y_hat_sample)

        fig_hysteresis = plot_hysteresis(x_inv, y_inv, y_hat_inv)
        self._log_figure(trainer, fig_hysteresis, tag=f"{tag_prefix}/hysteresis")
        plt.close(fig_hysteresis)

        density = output["density"][0].detach().cpu()

        if isinstance(pl_module, SelfAdaptivePreisach):
            alpha = pl_module.model.alpha.value
            beta = pl_module.model.beta.value
        elif isinstance(pl_module, DifferentiablePreisach | DifferentiablePreisachNN):
            alpha = pl_module.model.mesh[:, 1]
            beta = pl_module.model.mesh[:, 0]
        elif isinstance(pl_module, EncoderDecoderPreisachNN):
            alpha = pl_module.model.base_mesh[:, 1]
            beta = pl_module.model.base_mesh[:, 0]
        else:
            msg = "Model not supported"
            log.error(msg)
            return

        alpha = alpha.detach().cpu()
        beta = beta.detach().cpu()

        fig_density = plot_hysteron_density(alpha, beta, density)
        self._log_figure(trainer, fig_density, tag=f"{tag_prefix}/hysteron_density")
        plt.close(fig_density)

        if isinstance(pl_module, EncoderDecoderPreisachNN):
            initial_states = output["initial_states"][0].detach().cpu()
            mesh_coords = output["mesh_coords"][0].detach().cpu()
            alpha_perturbed = mesh_coords[:, 1]
            beta_perturbed = mesh_coords[:, 0]

            fig_initial_states = plot_initial_states(
                alpha_perturbed,
                beta_perturbed,
                initial_states,
            )

            self._log_figure(
                trainer, fig_initial_states, tag=f"{tag_prefix}/initial_states"
            )
            plt.close(fig_initial_states)

        if self.hysteron_scatter:
            fig_scatter = plot_hysteron_scatter(alpha, beta)
            self._log_figure(trainer, fig_scatter, tag=f"{tag_prefix}/hysteron_scatter")
            plt.close(fig_scatter)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: SelfAdaptivePreisach
        | DifferentiablePreisach
        | DifferentiablePreisachNN
        | EncoderDecoderPreisachNN,
        outputs: dict[str, torch.Tensor],
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return

        if not self.plot_training:
            return

        if trainer.global_step % self.train_plot_interval != 0:
            return

        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        dataset = typing.cast(
            TimeSeriesDataset | EncoderDecoderDataset, train_dataloader.dataset
        )

        h_transform, b_transform = list(dataset.transforms.values())

        pl_module.eval()
        with torch.no_grad():
            out = pl_module.common_step(batch, batch_idx)

        x_sample = out["x"][0]
        y_sample = out["y"][0]
        y_hat_sample = out["y_hat"][0]

        if x_sample.ndim > 1 and x_sample.shape[-1] > 1:
            x_sample = x_sample[..., 0]

        x_sample = x_sample.detach().cpu()
        y_sample = y_sample.detach().cpu()
        y_hat_sample = y_hat_sample.detach().cpu()

        x_inv = h_transform.inverse_transform(x_sample)
        y_inv = b_transform.inverse_transform(x_sample, y_sample)
        y_hat_inv = b_transform.inverse_transform(x_sample, y_hat_sample)

        fig_hysteresis = plot_hysteresis(x_inv, y_inv, y_hat_inv)
        self._log_figure(trainer, fig_hysteresis, tag="train/hysteresis")
        plt.close(fig_hysteresis)

        if isinstance(pl_module, EncoderDecoderPreisachNN):
            initial_states = out["initial_states"][0].detach().cpu()

            mesh_coords = out["mesh_coords"][0].detach().cpu()
            alpha_perturbed = mesh_coords[:, 1]
            beta_perturbed = mesh_coords[:, 0]

            fig_initial_states = plot_initial_states(
                alpha_perturbed,
                beta_perturbed,
                initial_states,
            )

            self._log_figure(trainer, fig_initial_states, tag="train/initial_states")
            plt.close(fig_initial_states)

        pl_module.train()

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

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    y_hat_np = y_hat.detach().cpu().numpy()

    data_color = plt.cm.tab10(0)
    model_color = plt.cm.tab10(1)

    ax1.plot(x_np, y_np, color=data_color, label="data", alpha=0.7)
    ax1.plot(x_np, y_hat_np, color=model_color, label="model", alpha=0.7)

    ax2.plot(x_np, y_np - y_hat_np, color=data_color, alpha=0.7)

    ax1.set_xlabel("I [A]")
    ax1.set_ylabel("B [T]")

    ax1.legend()
    ax1.grid()
    ax1.set_title("Hysteresis")

    ax2.set_xlabel("I [A]")
    ax2.set_ylabel("B [T]")
    ax2.set_title("Error")
    ax2.grid()

    fig.tight_layout()

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

    fig.tight_layout()

    return fig


def plot_initial_states(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_states: torch.Tensor,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    initial_states = initial_states.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    beta = beta.detach().cpu().numpy()

    c = ax.tripcolor(
        beta,
        alpha,
        initial_states,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$\\alpha$")

    ax.set_title("Initial hysteron states")

    fig.tight_layout()

    return fig


def plot_hysteron_scatter(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    alpha = alpha.detach().cpu().numpy()
    beta = beta.detach().cpu().numpy()

    # Adjust marker size based on number of mesh points
    n_points = len(alpha)
    # Heuristic: scale inversely with sqrt(n_points), clamped between 0.1 and 10
    marker_size = max(0.1, min(10, 100 / (n_points**0.5)))

    ax.scatter(
        beta,
        alpha,
        s=marker_size,
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

    fig.tight_layout()

    return fig
