"""Anchor-row geometry: the x <-> grid conversions must round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from utils.anchor import (
    build_anchor_rows,
    expectation_to_x,
    grid_to_x,
    interp_polyline_at_ys,
    x_to_grid,
)


def test_anchor_rows_span_config_range():
    ys = build_anchor_rows(56, 0.42, 1.0, 288)
    assert ys.shape == (56,)
    assert ys[0] == pytest.approx(0.42 * 288, abs=1e-4)
    assert ys[-1] == pytest.approx(288.0, abs=1e-4)
    assert np.all(np.diff(ys) > 0)


def test_x_to_grid_roundtrip_stays_inside_cell():
    xs = np.array([0.0, 4.0, 399.9, 799.9], dtype=np.float32)
    grids = x_to_grid(xs, img_w=800, num_grid=100)
    back = grid_to_x(grids, img_w=800, num_grid=100)
    # grid_to_x returns the cell centre, so it must stay within half a cell (4 px)
    assert np.all(np.abs(back - xs) <= 4.0 + 1e-3)
    assert grids.min() >= 0 and grids.max() <= 99


def test_x_to_grid_clips_out_of_range():
    assert x_to_grid(np.array([-50.0]), 800, 100)[0] == 0
    assert x_to_grid(np.array([900.0]), 800, 100)[0] == 99


def test_expectation_to_x_matches_grid_centre_when_one_hot():
    probs = np.zeros((1, 100), dtype=np.float32)
    probs[0, 50] = 1.0
    x = expectation_to_x(probs, img_w=800, num_grid=100)
    assert x[0] == pytest.approx((50 + 0.5) * 8.0)


def test_interp_polyline_interpolates_inside_and_nans_outside():
    poly = np.array([[10.0, 0.0], [20.0, 10.0]], dtype=np.float32)
    xs = interp_polyline_at_ys(poly, np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float32))
    assert xs[0] == pytest.approx(10.0)
    assert xs[1] == pytest.approx(15.0)
    assert xs[2] == pytest.approx(20.0)
    assert np.isnan(xs[3])


def test_interp_polyline_handles_unsorted_and_degenerate_input():
    poly = np.array([[20.0, 10.0], [10.0, 0.0]], dtype=np.float32)  # unsorted in y
    xs = interp_polyline_at_ys(poly, np.array([5.0], dtype=np.float32))
    assert xs[0] == pytest.approx(15.0)

    too_few = np.array([[1.0, 1.0]], dtype=np.float32)
    assert np.all(np.isnan(interp_polyline_at_ys(too_few, np.array([1.0], dtype=np.float32))))

    flat = np.array([[1.0, 5.0], [2.0, 5.0]], dtype=np.float32)  # single distinct y
    assert np.all(np.isnan(interp_polyline_at_ys(flat, np.array([5.0], dtype=np.float32))))
