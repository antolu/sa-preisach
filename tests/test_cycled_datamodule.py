from __future__ import annotations

import numpy as np
import pytest
import torch

from sa_preisach.data._cycled_datamodule import (
    CycleBatch,
    CycleDataset,
    cycle_collate_fn,
)

import pandas as pd


def _make_dfs(lengths: list[int]) -> list[pd.DataFrame]:
    return [
        pd.DataFrame({"h": np.random.rand(n), "b": np.random.rand(n)})
        for n in lengths
    ]


def test_cycle_dataset_len() -> None:
    dfs = _make_dfs([10, 20, 5])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    assert len(ds) == 3


def test_cycle_dataset_getitem_shapes() -> None:
    dfs = _make_dfs([10])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    item = ds[0]
    assert item["input"].shape == (10, 1)
    assert item["target"].shape == (10, 1)
    assert item["lengths"].item() == 10


def test_cycle_dataset_lengths_scalar_long() -> None:
    dfs = _make_dfs([7])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    item = ds[0]
    assert item["lengths"].dtype == torch.long
    assert item["lengths"].ndim == 0


def test_cycle_dataset_multi_feature_input() -> None:
    dfs = [pd.DataFrame({"h": np.random.rand(8), "extra": np.random.rand(8), "b": np.random.rand(8)})]
    ds = CycleDataset(dfs, input_cols=["h", "extra"], target_col="b")
    item = ds[0]
    assert item["input"].shape == (8, 2)


def test_cycle_dataset_transforms_property_empty_by_default() -> None:
    dfs = _make_dfs([5])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    assert ds.transforms == {}


def test_cycle_dataset_transforms_property_returns_copy() -> None:
    from unittest.mock import MagicMock
    from transformertf.data.transform import BaseTransform

    mock_transform = MagicMock(spec=BaseTransform)
    dfs = _make_dfs([5])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b", transforms={"h": mock_transform})
    t1 = ds.transforms
    t2 = ds.transforms
    assert t1 is not t2
    assert t1["h"] is mock_transform


def test_cycle_collate_fn_uniform_lengths() -> None:
    dfs = _make_dfs([6, 6])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    batch = cycle_collate_fn([ds[0], ds[1]])
    assert batch["input"].shape == (2, 6, 1)
    assert batch["target"].shape == (2, 6, 1)
    assert batch["lengths"].tolist() == [6, 6]


def test_cycle_collate_fn_variable_lengths_pads_to_max() -> None:
    dfs = _make_dfs([3, 7])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    batch = cycle_collate_fn([ds[0], ds[1]])
    assert batch["input"].shape == (2, 7, 1)
    assert batch["target"].shape == (2, 7, 1)
    assert batch["lengths"].tolist() == [3, 7]


def test_cycle_collate_fn_padding_is_zero() -> None:
    dfs = _make_dfs([2, 5])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    batch = cycle_collate_fn([ds[0], ds[1]])
    # Positions [2:5] of the short sequence should be zero-padded
    assert (batch["input"][0, 2:, :] == 0.0).all()
    assert (batch["target"][0, 2:, :] == 0.0).all()


def test_cycle_collate_fn_valid_region_unchanged() -> None:
    df = pd.DataFrame({"h": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]})
    ds = CycleDataset([df, df], input_cols=["h"], target_col="b")
    batch = cycle_collate_fn([ds[0], ds[1]])
    assert torch.allclose(batch["input"][0, :3, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(batch["target"][0, :3, 0], torch.tensor([0.4, 0.5, 0.6]))


def test_cycle_collate_fn_lengths_dtype() -> None:
    dfs = _make_dfs([4, 6])
    ds = CycleDataset(dfs, input_cols=["h"], target_col="b")
    batch = cycle_collate_fn([ds[0], ds[1]])
    assert batch["lengths"].dtype == torch.long
