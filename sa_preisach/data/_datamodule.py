from __future__ import annotations

import typing

import pandas as pd
from transformertf.data.dataset import TimeSeriesDataset
from transformertf.data.transform import MinMaxScaler, TransformCollection

if typing.TYPE_CHECKING:
    from transformertf.data._downsample import DOWNSAMPLE_METHODS
    from transformertf.data._dtype import VALID_DTYPES
    from transformertf.data.transform import BaseTransform


from transformertf.data import TimeSeriesDataModule


class PreisachDataModule(TimeSeriesDataModule):
    def __init__(  # noqa: PLR0913
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        dtype: VALID_DTYPES = "float32",
    ) -> None:
        if isinstance(known_covariates, str):
            known_covariates = [known_covariates]
        elif len(list(known_covariates)) > 1:
            msg = "known_covariates must be a single column name"
            raise ValueError(msg)

        # make sure that we have only 1 training dataset and 1 validation dataset
        if isinstance(train_df_paths, list) and len(train_df_paths) > 1:
            msg = "train_df_paths must be a single path"
            raise ValueError(msg)
        if isinstance(val_df_paths, list) and len(val_df_paths) > 1:
            msg = "val_df_paths must be a single path"
            raise ValueError(msg)

        super().__init__(
            known_covariates=known_covariates,
            target_covariate=target_covariate,
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            normalize=False,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            seq_len=0,
            extra_transforms=extra_transforms,
            batch_size=1,
            num_workers=0,
            dtype=dtype,
            shuffle=False,
            distributed=False,
        )
        self.save_hyperparameters(ignore=["extra_transforms"])

    def _make_dataset_from_df(
        self, df: pd.DataFrame | list[pd.DataFrame], *, predict: bool = False
    ) -> TimeSeriesDataset:
        if len(self.known_past_covariates) > 0:
            msg = "known_past_covariates is not used in this class."
            raise NotImplementedError(msg)

        input_cols = [cov.col for cov in self.known_covariates]
        target_data: pd.Series | list[pd.Series] | None
        if isinstance(df, pd.DataFrame):
            if self.target_covariate.col in df.columns:
                target_data = df[self.target_covariate.col]
            else:
                target_data = None
        elif self.target_covariate.col in df[0].columns:
            target_data = [d[self.target_covariate.col] for d in df]
        else:
            target_data = None

        breakpoint()

        return TimeSeriesDataset(
            input_data=df[input_cols]
            if isinstance(df, pd.DataFrame)
            else [df[input_cols] for df in df],
            target_data=target_data,
            stride=1,
            seq_len=len(df) if isinstance(df, pd.DataFrame) else len(df[0]),
            predict=predict,
            transforms=self.transforms,
            dtype=self.hparams["dtype"],
        )

    def _create_transforms(self) -> None:
        """
        Add normalization between 0 and 1.
        """
        super()._create_transforms()

        # add normalization transform with minmax scaler
        self._transforms[self.known_covariates[0].name] = TransformCollection(
            *[
                [
                    *list(self.transforms[self.known_covariates[0].name].transforms),
                    MinMaxScaler(min_=0.0, max_=1.0),
                ]
            ],
        )
        self._transforms[self.target_covariate.name] = TransformCollection(
            *list(self.transforms[self.target_covariate.name].transforms),
            MinMaxScaler(
                min_=0.0,
                max_=1.0,
            ),
        )
