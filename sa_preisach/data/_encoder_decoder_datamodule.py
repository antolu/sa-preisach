from __future__ import annotations

import typing

from transformertf.data.datamodule import TransformerDataModule

if typing.TYPE_CHECKING:
    from transformertf.data._downsample import DOWNSAMPLE_METHODS
    from transformertf.data._dtype import VALID_DTYPES
    from transformertf.data.transform import BaseTransform


class EncoderDecoderPreisachDataModule(TransformerDataModule):
    """
    Data module for encoder-decoder Preisach neural network models.
    
    This class extends TransformerDataModule to provide encoder-decoder data handling
    specifically designed for Preisach hysteresis modeling. It supports:
    
    - Context sequences (encoder input) for learning initial hysteron states
    - Target sequences (decoder input/output) for magnetization prediction
    - Automatic sequence length management and batching
    - Integration with existing transform pipeline
    
    Parameters
    ----------
    known_covariates : str or sequence of str
        Column names for features known in both past and future contexts.
        For Preisach models, this is typically the magnetic field (H).
    target_covariate : str
        Column name for the target variable to predict (typically magnetization B).
    known_past_covariates : str or sequence of str, optional
        Column names for features known only in the past context.
        Can include additional historical features if available.
    ctxt_seq_len : int, optional
        Length of the context (encoder) sequence for learning initial states.
        Default is 500.
    tgt_seq_len : int, optional
        Length of the target (decoder) sequence for prediction. Default is 200.
    min_ctxt_seq_len : int, optional
        Minimum context sequence length for randomization. Default is None.
    min_tgt_seq_len : int, optional
        Minimum target sequence length for randomization. Default is None.
    randomize_seq_len : bool, optional
        Whether to randomize sequence lengths during training. Default is False.
    train_df_paths : str or list of str, optional
        Path(s) to training data files (Parquet format).
    val_df_paths : str or list of str, optional
        Path(s) to validation data files (Parquet format).
    downsample : int, optional
        Downsampling factor. Default is 1.
    downsample_method : DOWNSAMPLE_METHODS, optional
        Downsampling method. Default is "interval".
    target_depends_on : str, optional
        Column dependency for target transforms. Default is None.
    extra_transforms : dict, optional
        Additional transforms for data preprocessing. Default is None.
    dtype : VALID_DTYPES, optional
        Data type for tensors. Default is "float32".
    batch_size : int, optional
        Training batch size. Default is 32.
    num_workers : int, optional
        Number of data loading workers. Default is 4.
    shuffle : bool, optional
        Whether to shuffle training data. Default is True.
    
    Notes
    -----
    This data module is designed specifically for the EncoderDecoderPreisachNN model.
    It provides the necessary data structure with:
    
    - `encoder_input`: Historical H/B sequences [batch_size, ctxt_seq_len, features]
    - `decoder_input`: Current H sequences [batch_size, tgt_seq_len, 1]
    - `target`: Target B sequences [batch_size, tgt_seq_len, 1]
    
    The encoder learns to map from historical behavior to appropriate initial
    hysteron states, while the decoder performs the actual magnetization prediction.
    
    Examples
    --------
    Basic usage for encoder-decoder Preisach training:
    
    >>> dm = EncoderDecoderPreisachDataModule(
    ...     known_covariates="I_ref_A",
    ...     target_covariate="B_meas_T_filtered", 
    ...     ctxt_seq_len=800,
    ...     tgt_seq_len=200,
    ...     train_df_paths="sample_train.parquet",
    ...     val_df_paths="sample_val.parquet",
    ...     batch_size=16
    ... )
    
    With sequence length randomization:
    
    >>> dm = EncoderDecoderPreisachDataModule(
    ...     known_covariates="I_ref_A",
    ...     target_covariate="B_meas_T_filtered",
    ...     ctxt_seq_len=1000,
    ...     tgt_seq_len=300,
    ...     min_ctxt_seq_len=500,
    ...     min_tgt_seq_len=150,
    ...     randomize_seq_len=True,
    ...     train_df_paths="sample_train.parquet",
    ...     val_df_paths="sample_val.parquet"
    ... )
    """
    
    def __init__(  # noqa: PLR0913
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        known_past_covariates: str | typing.Sequence[str] | None = None,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 200,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        dtype: VALID_DTYPES = "float32",
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> None:
        # Ensure known_covariates is a list
        if isinstance(known_covariates, str):
            known_covariates = [known_covariates]
        
        # For Preisach models, we typically want to add target to past context
        # so the encoder can see both H and B in the historical sequence
        add_target_to_past = True
        
        super().__init__(
            known_covariates=known_covariates,
            target_covariate=target_covariate,
            known_past_covariates=known_past_covariates,
            normalize=False,  # Disable normalization (handled in transforms)
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            min_ctxt_seq_len=min_ctxt_seq_len,
            min_tgt_seq_len=min_tgt_seq_len,
            randomize_seq_len=randomize_seq_len,
            stride=1,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            add_target_to_past=add_target_to_past,
            extra_transforms=extra_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            shuffle=shuffle,
            distributed=False,
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
        )
        
        self.save_hyperparameters(ignore=["extra_transforms"])
    
    @property
    def sequence_features(self) -> int:
        """
        Number of features in the encoder input sequence.
        
        For Preisach models with add_target_to_past=True, this includes
        both the known covariates and the target variable.
        
        Returns
        -------
        int
            Number of sequence features for encoder input
        """
        n_known = len(self.known_covariates) 
        n_target = 1 if self.hparams.get("add_target_to_past", True) else 0
        n_past = len(self.known_past_covariates) if self.known_past_covariates else 0
        
        return n_known + n_target + n_past