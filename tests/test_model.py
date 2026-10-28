"""Model wiring: shapes, aux-head gating, and the parameter count claimed in the README."""

from __future__ import annotations

import pytest
import torch

from models.ufld_rail import build_model
from utils.config import load_cfg


@pytest.fixture(scope="module")
def cfg():
    c = load_cfg("configs/default.yaml")
    c["model"]["pretrained"] = False  # no network access needed
    return c


@pytest.fixture(scope="module")
def dummy_input(cfg):
    return torch.randn(1, 3, cfg["input"]["height"], cfg["input"]["width"])


def test_forward_shapes_and_aux_gating(cfg, dummy_input):
    model = build_model(cfg)
    n_rows, n_grid = cfg["anchors"]["num_rows"], cfg["anchors"]["num_grid"]

    model.train()
    with torch.no_grad():
        logits, aux = model(dummy_input)
    assert logits.shape == (1, 2, n_rows, n_grid + 1)
    assert aux is not None and aux.shape == (1, 1, cfg["input"]["height"], cfg["input"]["width"])

    model.eval()
    with torch.no_grad():
        logits_eval, aux_eval = model(dummy_input)
    assert logits_eval.shape == logits.shape
    assert aux_eval is None  # aux head is training-only


def test_aux_head_can_be_disabled(cfg, dummy_input):
    c = {**cfg, "model": {**cfg["model"], "use_aux_seg": False}}
    model = build_model(c).train()
    with torch.no_grad():
        logits, aux = model(dummy_input)
    assert logits.shape[0] == 1
    assert aux is None


def test_resnet18_parameter_count_matches_readme(cfg):
    model = build_model(cfg)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert 14.0 < n < 15.0, f"expected ~14.6M params, got {n:.2f}M"


def test_mobilenet_backbone_builds_and_runs(cfg):
    c = {**cfg, "model": {**cfg["model"], "backbone": "mobilenet_v3_small"}}
    model = build_model(c).eval()
    with torch.no_grad():
        logits, _ = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 2, cfg["anchors"]["num_rows"], cfg["anchors"]["num_grid"] + 1)


def test_smaller_input_is_handled_by_adaptive_pooling(cfg):
    model = build_model(cfg).eval()
    with torch.no_grad():
        logits, _ = model(torch.randn(1, 3, 96, 320))
    assert logits.shape == (1, 2, cfg["anchors"]["num_rows"], cfg["anchors"]["num_grid"] + 1)
