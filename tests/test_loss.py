"""Loss composition, and the seg-masking behaviour that replaced the zero-filled masks."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from losses import RowAnchorLoss

NUM_GRID, NUM_ROWS = 20, 6


def _tensors(b: int = 2):
    torch.manual_seed(0)
    logits = torch.randn(b, 2, NUM_ROWS, NUM_GRID + 1)
    target = torch.randint(0, NUM_GRID, (b, 2, NUM_ROWS))
    seg_logits = torch.randn(b, 1, 8, 8)
    seg_gt = (torch.rand(b, 8, 8) > 0.5).float()
    return logits, target, seg_logits, seg_gt


def test_returns_finite_scalars():
    loss_fn = RowAnchorLoss(NUM_GRID)
    out = loss_fn(*_tensors(), seg_valid=torch.ones(2))
    for key in ("total", "cls", "sim", "shp", "seg"):
        assert key in out
        assert torch.isfinite(out[key]).all()
    assert float(out["total"]) > 0


def test_all_invalid_seg_samples_contribute_nothing():
    loss_fn = RowAnchorLoss(NUM_GRID)
    out = loss_fn(*_tensors(), seg_valid=torch.zeros(2))
    assert float(out["seg"]) == 0.0
    # and the total still reduces to the other terms
    assert torch.isfinite(out["total"])


def test_seg_loss_averages_only_valid_samples():
    loss_fn = RowAnchorLoss(NUM_GRID)
    logits, target, seg_logits, seg_gt = _tensors()
    out = loss_fn(logits, target, seg_logits, seg_gt, seg_valid=torch.tensor([1.0, 0.0]))
    expected = F.binary_cross_entropy_with_logits(seg_logits[:1].squeeze(1), seg_gt[:1])
    assert float(out["seg"]) == pytest.approx(float(expected), rel=1e-5)


def test_no_seg_inputs_means_zero_seg_term():
    loss_fn = RowAnchorLoss(NUM_GRID)
    logits, target, _, _ = _tensors()
    out = loss_fn(logits, target)
    assert float(out["seg"]) == 0.0


def test_lambda_weights_are_respected():
    logits, target, seg_logits, seg_gt = _tensors()
    plain = RowAnchorLoss(NUM_GRID)(logits, target, seg_logits, seg_gt)
    no_seg = RowAnchorLoss(NUM_GRID, lambda_seg=0.0)(logits, target, seg_logits, seg_gt)
    assert float(no_seg["total"]) == pytest.approx(float(plain["total"] - plain["seg"]), rel=1e-6)
