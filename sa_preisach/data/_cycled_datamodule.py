from __future__ import annotations

import typing

import pandas as pd
import torch
import torch.utils.data
from transformertf.data.datamodule._base import DataModuleBase
from transformertf.data.transform import BaseTransform

from .._mod_replace import replace_modname

if typing.TYPE_CHECKING:
    from transformertf.data._downsample import DOWNSAMPLE_METHODS
    from transformertf.data._dtype import VALID_DTYPES
    from transformertf.data.transform import BaseTransform


class CycleBatch(typing.TypedDict, total=False):
    input: torch.Tensor    # [batch, seq_len, n_features]  feature 0 = H_norm
    target: torch.Tensor   # [batch, seq_len, 1]           B_norm
    lengths: torch.Tensor  # [batch]                       valid timesteps per sample


class CycleDataset(torch.utils.data.Dataset):
    """
    Dataset where each item is one complete cycle.

    Parameters
    ----------
    dfs:
        List of preprocessed DataFrames, one per cycle.
    input_cols:
        Column names to stack as the input tensor (feature dimension).
    target_col:
        Column name for the target tensor.
    transforms:
        Mapping of column-name → transform, forwarded from the datamodule.
        Exposed via the ``transforms`` property so callbacks can call
        ``inverse_transform`` for plotting.
    dtype:
        Tensor dtype string, e.g. "float32".
    """

    def __init__(
        self,
        dfs: list[pd.DataFrame],
        input_cols: list[str],
        target_col: str,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: str = "float32",
    ) -> None:
        self._dfs = dfs
        self._input_cols = input_cols
        self._target_col = target_col
        self._transforms: dict[str, BaseTransform] = transforms or {}
        self._torch_dtype = getattr(torch, dtype)

    @property
    def transforms(self) -> dict[str, BaseTransform]:
        return dict(self._transforms.items())

    def __len__(self) -> int:
        return len(self._dfs)

    def __getitem__(self, idx: int) -> CycleBatch:
        df = self._dfs[idx]
        inp = torch.tensor(
            df[self._input_cols].to_numpy(), dtype=self._torch_dtype
        )  # [T, n_features]
        target = torch.tensor(
            df[[self._target_col]].to_numpy(), dtype=self._torch_dtype
        )  # [T, 1]
        length = torch.tensor(inp.shape[0], dtype=torch.long)
        return typing.cast(CycleBatch, {"input": inp, "target": target, "lengths": length})


def cycle_collate_fn(
    samples: list[CycleBatch],
) -> CycleBatch:
    """
    Collate variable-length cycle samples into a padded batch.

    Pads input and target to the length of the longest cycle in the batch.
    Padding value is 0.0. ``lengths`` is stacked from each sample.
    """
    inputs = [s["input"] for s in samples]
    targets = [s["target"] for s in samples]
    lengths = torch.stack([s["lengths"] for s in samples])  # [batch]

    padded_input = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=0.0
    )  # [batch, max_T, n_features]
    padded_target = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0.0
    )  # [batch, max_T, 1]

    return typing.cast(
        CycleBatch,
        {"input": padded_input, "target": padded_target, "lengths": lengths},
    )


class CycledPreisachDataModule(DataModuleBase):
    """
    DataModule for the encoder-free :class:`PreisachNN` model.

    Each file in ``train_df_paths`` / ``val_df_paths`` represents one complete
    excitation cycle.  Unlike the sliding-window :class:`PreisachDataModule`,
    no windowing is applied — each cycle becomes a single dataset item whose
    length equals the number of samples in the file.

    Cycles of different lengths are handled by the custom :func:`cycle_collate_fn`
    which zero-pads to the longest cycle in each batch and emits a ``lengths``
    tensor.  All validation cycles are merged into a single dataloader (not one
    per file).

    Only one ``known_covariate`` is accepted (the excitation current / applied
    field H).

    Parameters
    ----------
    known_covariate:
        Name of the single input column (H / excitation current).
    target_covariate:
        Name of the target column (B / flux density).
    train_df_paths:
        Paths to training cycle files.
    val_df_paths:
        Paths to validation cycle files.
    batch_size:
        Number of cycles per training batch.
    val_batch_size:
        Number of cycles per validation batch.
    shuffle:
        Whether to shuffle training cycles across epochs.
    downsample:
        Downsample factor applied uniformly to every cycle.
    downsample_method:
        Downsampling strategy.
    extra_transforms:
        Additional per-column transforms, forwarded to :class:`DataModuleBase`.
    dtype:
        Tensor dtype.
    distributed:
        Distributed-sampler configuration.
    """

    def __init__(
        self,
        *,
        known_covariate: str,
        target_covariate: str,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        batch_size: int = 8,
        val_batch_size: int = 8,
        shuffle: bool = True,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        dtype: VALID_DTYPES = "float32",
        distributed: bool | typing.Literal["auto"] = "auto",
    ) -> None:
        super().__init__(
            known_covariates=[known_covariate],
            target_covariate=target_covariate,
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            normalize=False,
            downsample=downsample,
            downsample_method=downsample_method,
            extra_transforms=extra_transforms,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=0,
            dtype=dtype,
            shuffle=shuffle,
            distributed=distributed,
        )
        self.save_hyperparameters(ignore=["extra_transforms"])

    def _make_dataset_from_df(
        self,
        df: pd.DataFrame | list[pd.DataFrame],
        *,
        predict: bool = False,
    ) -> CycleDataset:
        dfs = [df] if isinstance(df, pd.DataFrame) else df
        input_cols = [cov.col for cov in self.known_covariates]
        target_col = self.target_covariate.col
        return CycleDataset(
            dfs=dfs,
            input_cols=input_cols,
            target_col=target_col,
            transforms=self.transforms,
            dtype=self.hparams["dtype"],
        )

    def collate_fn(  # type: ignore[override]
        self,
    ) -> typing.Callable[[list[CycleBatch]], CycleBatch]:
        return cycle_collate_fn

    @property
    def train_dataset(self) -> CycleDataset:
        return self._make_dataset_from_df(self._train_df)

    @property
    def val_dataset(self) -> CycleDataset:
        return self._make_dataset_from_df(self._val_df)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        sampler: torch.utils.data.Sampler | None = None
        if self.distributed_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                shuffle=self.hparams["shuffle"],
                drop_last=True,
            )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=sampler is None and self.hparams["shuffle"],
            num_workers=0,
            sampler=sampler,
            pin_memory=True,
            collate_fn=self.collate_fn(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        sampler: torch.utils.data.Sampler | None = None
        if self.distributed_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset,
                shuffle=False,
                drop_last=False,
            )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams["val_batch_size"],
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            pin_memory=True,
            collate_fn=self.collate_fn(),
        )


replace_modname(CycledPreisachDataModule, __name__)
