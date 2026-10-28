"""Crop parsing/geometry and end-to-end inference on a synthetic video."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from infer_video import apply_crop, infer_frame, parse_crop, resolve_device
from models.ufld_rail import build_model
from utils.config import load_cfg


def test_parse_crop_none_and_valid():
    assert parse_crop(None) is None
    assert parse_crop("0.1, 0.9, 0.2, 0.8") == pytest.approx((0.1, 0.9, 0.2, 0.8))


@pytest.mark.parametrize(
    "spec, match",
    [
        ("0.1,0.9,0.2", "4 comma-separated values"),
        ("a,b,c,d", "must be numbers"),
        ("0.1,0.9,0.2,1.5", r"right=1.5 must be within"),
        ("0.9,0.1,0.2,0.8", "top < bottom"),
        ("0.1,0.9,0.8,0.2", "left < right"),
    ],
)
def test_parse_crop_rejects_bad_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_crop(spec)


def test_apply_crop_geometry():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi, (x0, y0, cw, ch) = apply_crop(frame, (0.1, 0.5, 0.2, 0.6))
    assert (x0, y0, cw, ch) == (40, 10, 80, 40)
    assert roi.shape == (40, 80, 3)


def test_apply_crop_rejects_degenerate_roi():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="degenerates"):
        apply_crop(frame, (0.5, 0.501, 0.0, 1.0))


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError):
        resolve_device("cuda")  # this machine has no CUDA


@pytest.fixture(scope="module")
def tiny_model():
    cfg = load_cfg("configs/default.yaml")
    cfg["model"]["pretrained"] = False
    cfg["input"] = {"width": 128, "height": 64}
    model = build_model(cfg).eval()
    return cfg, model


def test_infer_frame_returns_frame_space_polylines(tiny_model):
    cfg, model = tiny_model
    frame = np.random.randint(0, 255, (120, 320, 3), dtype=np.uint8)
    lp, rp, lv, rv = infer_frame(model, frame, cfg, torch.device("cpu"), conf=0.3)

    n = cfg["anchors"]["num_rows"]
    assert lp.shape == (n, 2) and rp.shape == (n, 2)
    assert lv.shape == (n,) and rv.shape == (n,)
    assert lp.dtype == np.float32
    # y grows downward and stays inside the frame
    assert np.all(np.diff(lp[:, 1]) > 0)
    assert lp[:, 1].min() >= 0 and lp[:, 1].max() <= frame.shape[0]
    assert lp[:, 0].min() >= 0 and lp[:, 0].max() <= frame.shape[1]


def test_infer_frame_with_crop_maps_back_to_full_frame_coordinates(tiny_model):
    cfg, model = tiny_model
    frame = np.random.randint(0, 255, (120, 320, 3), dtype=np.uint8)
    lp, rp, lv, rv = infer_frame(
        model, frame, cfg, torch.device("cpu"), conf=0.3, crop=(0.25, 0.75, 0.1, 0.9)
    )
    # crop starts at x0=32, y0=30 -> every y must be >= 30
    assert lp[:, 1].min() >= 30 - 1e-4
    assert lp[:, 0].min() >= 32 - 1e-4
