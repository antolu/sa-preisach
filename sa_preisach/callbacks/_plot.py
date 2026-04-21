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
from ..utils import get_states

log = logging.getLogger(__name__)


def _try_get_logger_class(attr: str) -> type | None:
    try:
        obj = L.pytorch.loggers
        for part in attr.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    else:
        return obj  # type: ignore[return-value]


class PlotHysteresisCallback(L.pytorch.callbacks.Callback):
    def __init__(
        self,
        *,
        validate_every_n_epochs: int = 1,
        hysteron_scatter: bool = True,
        plot_unscaled: bool = False,
        plot_training: bool = False,
        train_plot_interval: int = 100,
        num_samples: int = 1,
    ) -> None:
        super().__init__()
        self.validate_every_n_epochs = validate_every_n_epochs
        self.hysteron_scatter = hysteron_scatter
        self.plot_training = plot_training
        self.train_plot_interval = train_plot_interval
        self.num_samples = num_samples

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
                dataset = typing.cast(
                    TimeSeriesDataset | EncoderDecoderDataset, dataloader.dataset
                )
                hysteresis_output = self._stitched_rollout(pl_module, dataset, outputs)
                self._log_validation_output(
                    trainer=trainer,
                    pl_module=pl_module,
                    dataset=dataset,
                    output=outputs[0],
                    hysteresis_output=hysteresis_output,
                    tag_prefix=f"validation/{dataloader_idx}",
                )
        else:
            if not pl_module.validation_outputs:
                return super().on_validation_epoch_end(trainer, pl_module)
            dataset = typing.cast(
                TimeSeriesDataset | EncoderDecoderDataset, dataloaders.dataset
            )
            hysteresis_output = self._stitched_rollout(
                pl_module, dataset, pl_module.validation_outputs
            )
            self._log_validation_output(
                trainer=trainer,
                pl_module=pl_module,
                dataset=dataset,
                output=pl_module.validation_outputs[0],
                hysteresis_output=hysteresis_output,
                tag_prefix="validation",
            )

        return super().on_validation_epoch_end(trainer, pl_module)

    @staticmethod
    def _concat_outputs(
        outputs: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if len(outputs) == 1:
            return outputs[0]
        seq_keys = ("x", "y", "y_hat")
        merged: dict[str, torch.Tensor] = {}
        for key in seq_keys:
            if key in outputs[0]:
                merged[key] = torch.cat([o[key] for o in outputs], dim=1)
        for key in outputs[0]:
            if key not in merged:
                merged[key] = outputs[0][key]
        return merged

    def _stitched_rollout(
        self,
        pl_module: SelfAdaptivePreisach
        | DifferentiablePreisach
        | DifferentiablePreisachNN
        | EncoderDecoderPreisachNN,
        dataset: TimeSeriesDataset | EncoderDecoderDataset,
        outputs: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Build a continuous-rollout hysteresis output for the whole validation
        dataloader. For EncoderDecoderPreisachNN the encoder is run on the
        first window; subsequent windows reuse the previous window's terminal
        hysteron state via the model's ``initial_states`` override. This
        removes the stitching discontinuities that appear when independent
        per-batch outputs are concatenated.
        """
        if not isinstance(pl_module, EncoderDecoderPreisachNN):
            n = min(self.num_samples, len(outputs))
            return self._concat_outputs(outputs[:n])

        n_windows = min(self.num_samples, len(dataset))
        if n_windows <= 0:
            return outputs[0]

        was_training = pl_module.training
        pl_module.eval()

        device = next(pl_module.parameters()).device
        model = pl_module.model
        temp = pl_module.hparams["temp"]

        y_hat_chunks: list[torch.Tensor] = []
        y_chunks: list[torch.Tensor] = []
        x_chunks: list[torch.Tensor] = []
        prev_states: torch.Tensor | None = None
        prev_y0: torch.Tensor | None = None

        with torch.no_grad():
            for i in range(n_windows):
                sample = dataset[i]
                enc_in = sample["encoder_input"].unsqueeze(0).to(device)
                dec_in = sample["decoder_input"].unsqueeze(0).to(device)
                target = sample["target"].unsqueeze(0).to(device)

                y0 = enc_in[:, -1, 0] if i == 0 else prev_y0

                y_hat, density, _m, states_used, mesh_coords = model(
                    encoder_input=enc_in,
                    decoder_input=dec_in[..., 0:1],
                    y0=y0,
                    initial_states=prev_states,
                    temp=temp,
                )

                dec_len = int(sample["decoder_lengths"].item())
                alpha = mesh_coords[0, :, 1].cpu()
                beta = mesh_coords[0, :, 0].cpu()
                h_seq = dec_in[0, :dec_len, 0].cpu()
                assert y0 is not None
                states_seq = get_states(
                    h=h_seq,
                    alpha=alpha,
                    beta=beta,
                    current_state=states_used[0].cpu(),
                    current_field=y0[0].cpu(),
                    temp=temp,
                    dtype=torch.float32,
                    training=False,
                )
                prev_states = states_seq[-1].unsqueeze(0).to(device)
                prev_y0 = h_seq[-1].unsqueeze(0).to(device)

                y_hat_chunks.append(y_hat[0, :dec_len].cpu())
                y_chunks.append(target[0, :dec_len].squeeze(-1).cpu())
                x_chunks.append(dec_in[0, :dec_len, 0].cpu())

        if was_training:
            pl_module.train()

        def _cat_with_nan(chunks: list[torch.Tensor]) -> torch.Tensor:
            if len(chunks) == 1:
                return chunks[0]
            nan = torch.full((1,), float("nan"))
            parts = [c for chunk in chunks for c in (chunk, nan)][:-1]
            return torch.cat(parts)

        y_hat_full = _cat_with_nan(y_hat_chunks).unsqueeze(0)
        y_full = _cat_with_nan(y_chunks).unsqueeze(0)
        x_full = _cat_with_nan(x_chunks).unsqueeze(0)

        merged: dict[str, torch.Tensor] = {
            "x": x_full,
            "y": y_full,
            "y_hat": y_hat_full,
        }
        for key, value in outputs[0].items():
            if key not in merged:
                merged[key] = value
        return merged

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
        hysteresis_output: dict[str, torch.Tensor] | None = None,
        tag_prefix: str,
    ) -> None:
        h_transform, b_transform = list(dataset.transforms.values())

        hyst_out = hysteresis_output if hysteresis_output is not None else output
        x_sample = hyst_out["x"][0]
        y_sample = hyst_out["y"][0]
        y_hat_sample = hyst_out["y_hat"][0]

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
            n_samples = min(self.num_samples, output["initial_states"].shape[0])
            for i in range(n_samples):
                initial_states = output["initial_states"][i].detach().cpu()
                mesh_coords = output["mesh_coords"][i].detach().cpu()
                alpha_perturbed = mesh_coords[:, 1]
                beta_perturbed = mesh_coords[:, 0]

                fig_initial_states = plot_initial_states(
                    alpha_perturbed,
                    beta_perturbed,
                    initial_states,
                )
                self._log_figure(
                    trainer,
                    fig_initial_states,
                    tag=f"{tag_prefix}/initial_states_{i}",
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
            n_samples = min(self.num_samples, out["initial_states"].shape[0])
            for i in range(n_samples):
                initial_states = out["initial_states"][i].detach().cpu()
                mesh_coords = out["mesh_coords"][i].detach().cpu()
                alpha_perturbed = mesh_coords[:, 1]
                beta_perturbed = mesh_coords[:, 0]

                fig_initial_states = plot_initial_states(
                    alpha_perturbed,
                    beta_perturbed,
                    initial_states,
                )
                self._log_figure(
                    trainer, fig_initial_states, tag=f"train/initial_states_{i}"
                )
                plt.close(fig_initial_states)

        pl_module.train()

    def _log_figure(
        self, trainer: L.Trainer, fig: matplotlib.figure.Figure, tag: str
    ) -> None:
        if trainer.logger is None:
            return
        logger = trainer.logger
        if (
            neptune_cls := _try_get_logger_class("neptune.NeptuneLogger")
        ) and isinstance(logger, neptune_cls):
            typing.cast(L.pytorch.loggers.neptune.NeptuneLogger, logger).experiment[
                tag
            ].append(fig)
        elif (tb_cls := _try_get_logger_class("TensorBoardLogger")) and isinstance(
            logger, tb_cls
        ):
            typing.cast(
                L.pytorch.loggers.TensorBoardLogger, logger
            ).experiment.add_figure(tag, fig, global_step=trainer.global_step)
        elif (wandb_cls := _try_get_logger_class("WandbLogger")) and isinstance(
            logger, wandb_cls
        ):
            import wandb

            typing.cast(L.pytorch.loggers.WandbLogger, logger).experiment.log(
                {tag: wandb.Image(fig)}, commit=False
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
