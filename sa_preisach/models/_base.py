from __future__ import annotations

import collections
import typing

import lightning as L
import torch
from transformertf.data import TimeSeriesSample


class BaseModule(L.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.validation_outputs: list[dict[str, torch.Tensor]] = []

    def on_fit_start(self) -> None:
        self.maybe_compile_model()

        return super().on_fit_start()

    def maybe_compile_model(self) -> None:
        """
        Compile the model if the "compile_model" key is present in the hyperparameters
        and is set to True. This is up to the subclass to implement. This also
        requires the model to be set to the "model" attribute.
        """
        if self.hparams.get("compile_model"):
            for name, mod in self.named_children():
                if "loss" in name.lower():
                    continue
                setattr(self, name, torch.compile(mod))

    def on_train_epoch_start(self) -> None:
        if self.global_step > 0:
            return super().on_train_epoch_start()
            # check that we only have 1 training and validation batch
        if self.trainer.train_dataloader is None:
            msg = "No training dataloader found"
            raise ValueError(msg)

        if self.trainer.val_dataloaders is None:
            msg = "No validation dataloader found"
            raise ValueError(msg)

        if len(self.trainer.train_dataloader) != 1:
            msg = "Training dataloader must have only 1 batch"
            raise ValueError(msg)

        if isinstance(self.trainer.val_dataloaders, list):
            msg = "This model only supports a single validation dataloader"
            raise ValueError(msg)  # noqa: TRY004

        if len(self.trainer.val_dataloaders) != 1:
            msg = "Validation dataloader must have only 1 batch"
            raise ValueError(msg)

        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs.clear()
        return super().on_validation_epoch_start()

    def on_validation_batch_end(  # type: ignore[override]
        self,
        outputs: dict[str, torch.Tensor],
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.validation_outputs.append(outputs)

        return super().on_validation_batch_end(
            outputs, batch, batch_idx, dataloader_idx
        )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.hparams.get("log_grad_norm") and self.global_rank == 0:
            self.log_dict(L.pytorch.utilities.grad_norm(self, norm_type=2))

    def state_dict(
        self,
        *args: typing.Any,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # hack to save the original model state dict and not the compiled one
        # this assumes that internally the model is stored in the `model` attribute
        # and that the model is not compiled when the LightningModule is instantiated

        # keys are xxx._orig_mod.xxx, remove _orig_mod
        if self.hparams.get("compile_model"):
            odict = collections.OrderedDict()
            for k in list(state_dict.keys()):
                if "_orig_mod" in k:
                    new_key = k.replace("_orig_mod.", "")
                    odict[new_key] = state_dict[k]
                else:
                    odict[k] = state_dict[k]

            state_dict = odict

        return state_dict
