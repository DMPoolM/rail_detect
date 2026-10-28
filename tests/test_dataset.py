"""Dataset: id filtering, tensor contract, and the seg-validity flag."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets.railsem19 import RailSem19Dataset, split_ids


def test_list_ids_requires_a_matching_annotation(dataset_kwargs):
    root = dataset_kwargs["root"]
    ids = RailSem19Dataset.list_ids(root, "jpgs", "jsons")
    assert ids == ["rs19_val_00000", "rs19_val_00001", "rs19_val_00002"]

    # an image without its JSON must be filtered out, not crash a worker later
    import os

    os.remove(f"{root}/jsons/rs19_val_00001.json")
    assert RailSem19Dataset.list_ids(root, "jpgs", "jsons") == [
        "rs19_val_00000",
        "rs19_val_00002",
    ]


def test_missing_directory_is_reported(dataset_kwargs):
    with pytest.raises(FileNotFoundError, match="image directory not found"):
        RailSem19Dataset.list_ids(dataset_kwargs["root"], "nope", "jsons")


def test_item_contract(dataset_kwargs):
    ds = RailSem19Dataset(ids=["rs19_val_00000"], train=False, **dataset_kwargs)
    img, label, seg, seg_valid = ds[0]

    assert img.shape == (3, 32, 64) and img.dtype == torch.float32
    assert label.shape == (2, 8) and label.dtype == torch.int64
    assert int(label.min()) >= 0 and int(label.max()) <= 20
    assert seg.shape == (32, 64) and seg.dtype == torch.float32
    assert float(seg_valid) == 1.0
    assert set(seg.unique().tolist()) <= {0.0, 1.0}


def test_rail_rows_are_detected_for_a_centred_pair(dataset_kwargs):
    ds = RailSem19Dataset(ids=["rs19_val_00000"], train=False, **dataset_kwargs)
    _, label, _, _ = ds[0]
    # the fixture draws a centred pair reaching the bottom of the frame, so every
    # anchor row should carry both rails
    assert (label < 20).all()


def test_missing_label_map_is_flagged_instead_of_zero_filled(dataset_kwargs):
    ds = RailSem19Dataset(ids=["rs19_val_00002"], train=False, **dataset_kwargs)
    _, _, seg, seg_valid = ds[0]
    assert float(seg_valid) == 0.0
    assert float(seg.sum()) == 0.0


def test_hflip_swaps_rails_and_mirrors_grids(dataset_kwargs, monkeypatch):
    ds = RailSem19Dataset(ids=["rs19_val_00000"], train=False, **dataset_kwargs)
    label = np.array([[3, 4, 20], [10, 11, 20]], dtype=np.int64)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    seg = np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr("random.random", lambda: 0.0)  # force the flip branch
    _, _, out = ds._augment(img, seg, label)

    # mirrored grids (19 - g) with left/right swapped; "no rail" (20) stays 20
    assert out[0].tolist() == [9, 8, 20]
    assert out[1].tolist() == [16, 15, 20]


def test_split_ids_is_deterministic_and_disjoint():
    ids = [f"id{i}" for i in range(50)]
    tr1, va1 = split_ids(ids, 0.2, 7)
    tr2, va2 = split_ids(ids, 0.2, 7)
    assert (tr1, va1) == (tr2, va2)
    assert not set(tr1) & set(va1)
    assert len(va1) == 10 and len(tr1) == 40
