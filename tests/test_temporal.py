"""Temporal smoother: blends steady state, snaps on re-acquisition, never uses stale data."""

from __future__ import annotations

import numpy as np
import pytest

from utils.temporal import RailSmoother


def _pts(*xs):
    return np.array([[x, float(i)] for i, x in enumerate(xs)], dtype=np.float32)


def test_rejects_out_of_range_ema():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            RailSmoother(bad)


def test_ema_zero_is_a_pass_through():
    s = RailSmoother(0.0)
    p = _pts(1.0, 2.0)
    out = s.update(p, np.array([True, True]))
    assert np.allclose(out, p)


def test_first_frame_is_returned_unchanged():
    s = RailSmoother(0.9)
    p = _pts(5.0, 6.0)
    assert np.allclose(s.update(p, np.array([True, True])), p)


def test_blends_between_consecutive_frames():
    s = RailSmoother(0.5)
    s.update(_pts(0.0, 0.0), np.array([True, True]))
    out = s.update(_pts(10.0, 20.0), np.array([True, True]))
    assert out[0, 0] == pytest.approx(5.0)
    assert out[1, 0] == pytest.approx(10.0)


def test_reacquired_rows_snap_instead_of_drifting_to_stale_positions():
    s = RailSmoother(0.9)
    s.update(_pts(100.0, 100.0), np.array([True, True]))
    # rail disappears for a frame
    s.update(_pts(0.0, 0.0), np.array([False, False]))
    # ...and comes back far away: the old 100.0 must not leak into the result
    out = s.update(_pts(300.0, 300.0), np.array([True, True]))
    assert np.allclose(out[:, 0], 300.0)


def test_invalid_rows_are_never_blended():
    s = RailSmoother(0.9)
    s.update(_pts(10.0, 10.0), np.array([True, True]))
    out = s.update(_pts(50.0, 60.0), np.array([True, False]))
    assert out[0, 0] == pytest.approx(0.9 * 10.0 + 0.1 * 50.0)
    assert out[1, 0] == pytest.approx(60.0)


def test_reset_forgets_history():
    s = RailSmoother(0.9)
    s.update(_pts(100.0), np.array([True]))
    s.reset()
    assert np.allclose(s.update(_pts(7.0), np.array([True])), _pts(7.0))


def test_inputs_are_not_mutated():
    s = RailSmoother(0.5)
    p = _pts(1.0, 2.0)
    before = p.copy()
    s.update(p, np.array([True, True]))
    s.update(p * 3, np.array([True, True]))
    assert np.allclose(p, before)
