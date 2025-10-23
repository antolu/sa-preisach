from __future__ import annotations

import os
import pathlib
import sys
import typing
import warnings

import einops._torch_specific
import lightning as L
import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401
import torch
from lightning.pytorch.cli import LightningArgumentParser
from transformertf.data.datamodule import DataModuleBase
from transformertf.main import (
    NeptuneLoggerSaveConfigCallback,
    _FilterCallback,
    setup_logger,
)

from sa_preisach.data import PreisachDataModule
from sa_preisach.models import (  # noqa: F401
    BaseModule,
    DifferentiablePreisach,
    EncoderDecoderPreisachNN,
    SelfAdaptivePreisach,
)

warnings.filterwarnings("ignore", category=UserWarning)

einops._torch_specific.allow_ops_in_compiled_graph()  # noqa: SLF001


class LightningCLI(lightning.pytorch.cli.LightningCLI):
    model: SelfAdaptivePreisach
    datamodule: PreisachDataModule

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(parser_kwargs={"parser_mode": "omegaconf"}, **kwargs)

        # Dynamically set auto_configure_optimizers based on model type
        # Models that use manual optimization should disable auto configuration
        if (
            hasattr(self.model, "automatic_optimization")
            and not self.model.automatic_optimization
        ):
            self.auto_configure_optimizers = False
        else:
            # Enable auto configuration for models that use standard Lightning optimization
            self.auto_configure_optimizers = True

    def before_instantiate_classes(self) -> None:
        if hasattr(self.config, "fit") and hasattr(self.config.fit, "verbose"):
            setup_logger(self.config.fit.verbose)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "-v",
            dest="verbose",
            action="count",
            default=0,
            help="Verbose flag. Can be used more than once.",
        )

        parser.add_argument(
            "-n",
            "--experiment-name",
            dest="experiment_name",
            type=str,
            default=None,
            help="Name of the experiment.",
        )

        add_trainer_defaults(parser)

        add_callback_defaults(parser)

        parser.link_arguments(
            "data.n_train_samples",
            "model.init_args.n_train_samples",
            apply_on="instantiate",
        )

    def before_fit(self) -> None:  # noqa: PLR0912
        # hijack model checkpoint callbacks to save to checkpoint_dir/version_{version}
        if (
            hasattr(self.config, "fit")
            and hasattr(self.config.fit, "experiment_name")
            and self.config.fit.experiment_name
        ):
            logger_name = self.config.fit.experiment_name

            if isinstance(self.trainer.logger, L.pytorch.loggers.TensorBoardLogger):
                self.trainer.logger._name = logger_name  # noqa: SLF001
        else:
            logger_name = ""
        try:
            if isinstance(self.trainer.logger, L.pytorch.loggers.TensorBoardLogger):
                version = self.trainer.logger.version
            else:
                version = "na"
        except TypeError:
            version = 0
        version_str = f"version_{version}"

        # if logger is a neptune logger, save the config to a temporary file and upload it
        # also track artifacts from datamodule
        if isinstance(self.trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            import neptune

            # filter out errors caused by logging epoch more than once
            neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
                _FilterCallback()
            )

            if "train_df_paths" in self.datamodule.hparams:
                for train_df_path in self.datamodule.hparams["train_df_paths"]:
                    self.trainer.logger.experiment["train/dataset"].track_files(
                        os.fspath(pathlib.Path(train_df_path).expanduser())
                    )
            if "val_df_paths" in self.datamodule.hparams:
                if isinstance(self.datamodule.hparams["val_df_paths"], str):
                    self.trainer.logger.experiment["validation/dataset"].track_files(
                        os.fspath(
                            pathlib.Path(
                                self.datamodule.hparams["val_df_paths"]
                            ).expanduser()
                        )
                    )
                else:
                    for val_df_path in self.datamodule.hparams["val_df_paths"]:
                        self.trainer.logger.experiment[
                            "validation/dataset"
                        ].track_files(os.fspath(pathlib.Path(val_df_path).expanduser()))

            # log the command used to launch training to neptune
            self.trainer.logger.experiment["source_code/argv"] = " ".join(sys.argv)

            self.trainer.logger.experiment.sync()

        for callback in self.trainer.callbacks:
            if isinstance(callback, lightning.pytorch.callbacks.ModelCheckpoint):
                if logger_name:
                    dirpath = os.path.join(callback.dirpath, logger_name, version_str)
                else:
                    dirpath = os.path.join(callback.dirpath or ".", version_str)
                callback.dirpath = dirpath

        self.trainer.callbacks.append(
            NeptuneLoggerSaveConfigCallback(
                parser=self.parser, config=self.config, overwrite=True
            )
        )


def add_trainer_defaults(parser: LightningArgumentParser) -> None:
    parser.set_defaults({"trainer.use_distributed_sampler": False})


def add_callback_defaults(parser: LightningArgumentParser) -> None:
    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.LearningRateMonitor, "lr_monitor"
    )
    parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

    # parser.add_lightning_class_args(
    #     lightning.pytorch.callbacks.RichProgressBar, "progress_bar"
    # )
    # parser.set_defaults({"progress_bar.refresh_rate": 1})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.RichModelSummary, "model_summary"
    )
    parser.set_defaults({"model_summary.max_depth": 2})


def main() -> None:
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        model_class=BaseModule,
        datamodule_class=DataModuleBase,
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()
