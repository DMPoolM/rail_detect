"""RailSem19 JSON parsing and ego-pair selection, against the documented schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.railsem_parser import (
    load_rail_pairs,
    load_rail_polylines,
    read_meta,
    resolve_image_path,
    select_ego_rail_pair,
)

W, H = 100, 50


def _rail_pair(left_x0, left_x1, right_x0, right_x1):
    return {
        "label": "rail",
        "polyline-pair": [
            [[left_x0, 0], [left_x1, H - 1]],
            [[right_x0, 0], [right_x1, H - 1]],
        ],
    }


def _write(tmp_path: Path, name: str, objects, frame=None, sizes=True) -> Path:
    data = {"frame": frame or name, "objects": objects}
    if sizes:
        data["imgWidth"] = W
        data["imgHeight"] = H
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_ignores_guard_rail_and_rail_occluder(tmp_path):
    p = _write(
        tmp_path,
        "a",
        [
            {"label": "guard-rail", "polyline": [[0, 0], [W - 1, H - 1]]},
            {"label": "rail-occluder", "polygon": [[0, 0], [5, 0], [5, 5]]},
        ],
    )
    assert load_rail_polylines(p) == []
    assert load_rail_pairs(p) == []
    assert select_ego_rail_pair(p) is None


def test_prefers_the_pair_nearest_the_image_centre(tmp_path):
    centre = _rail_pair(30, 40, 70, 60)
    off_left = _rail_pair(2, 3, 6, 7)
    p = _write(tmp_path, "a", [off_left, centre])

    pair = select_ego_rail_pair(p)
    assert pair is not None
    # bottom ends of the centre pair: x=40 and x=60 -> midpoint 50 == W/2
    assert pair.left[-1][0] == 40
    assert pair.right[-1][0] == 60


def test_pair_indices_are_normalised_to_left_and_right(tmp_path):
    swapped = _rail_pair(70, 60, 30, 40)  # index 0 is actually the right-hand rail
    p = _write(tmp_path, "a", [swapped])
    pair = select_ego_rail_pair(p)
    assert pair.left[-1][0] == 40 and pair.right[-1][0] == 60


def test_pair_that_never_reaches_the_bottom_is_rejected(tmp_path):
    high = {
        "label": "rail",
        "polyline-pair": [
            [[30, 0], [40, 10]],
            [[70, 0], [60, 10]],
        ],
    }
    p = _write(tmp_path, "a", [high])
    assert select_ego_rail_pair(p) is None


def test_falls_back_to_loose_polylines_straddling_the_centre(tmp_path):
    objs = [
        {"label": "rail", "polyline": [[42, 0], [44, H - 1]]},
        {"label": "rail", "polyline": [[58, 0], [56, H - 1]]},
        {"label": "rail", "polyline": [[5, 0], [5, H - 1]]},
    ]
    p = _write(tmp_path, "a", objs)
    pair = select_ego_rail_pair(p)
    assert pair is not None
    assert pair.left[-1][0] == 44
    assert pair.right[-1][0] == 56


def test_dimensions_default_to_the_json_header(tmp_path):
    p = _write(tmp_path, "a", [_rail_pair(30, 40, 70, 60)])
    assert select_ego_rail_pair(p) is not None  # no explicit w/h passed


def test_missing_dimensions_and_no_header_is_an_error(tmp_path):
    p = _write(tmp_path, "a", [_rail_pair(30, 40, 70, 60)], sizes=False)
    with pytest.raises(ValueError, match="image size unknown"):
        select_ego_rail_pair(p)


def test_read_meta_and_frame_based_image_resolution(tmp_path):
    json_dir = tmp_path / "jsons"
    img_dir = tmp_path / "jpgs"
    json_dir.mkdir()
    img_dir.mkdir()
    p = _write(json_dir, "rs19_val_00042", [_rail_pair(30, 40, 70, 60)], frame="rs19_val_00042")

    meta = read_meta(p)
    assert meta == {"frame": "rs19_val_00042", "img_width": W, "img_height": H}

    # image on disk is named by `frame`, not by the json filename
    (img_dir / "rs19_val_00042.jpg").write_bytes(b"")
    assert resolve_image_path(p, img_dir).name == "rs19_val_00042.jpg"


def test_degenerate_polylines_are_dropped(tmp_path):
    p = _write(
        tmp_path,
        "a",
        [
            {"label": "rail", "polyline": [[10, 10]]},  # single point
            {"label": "rail", "polyline-pair": [[[1, 1]], [[2, 2]]]},  # both too short
        ],
    )
    assert load_rail_polylines(p) == []
    assert select_ego_rail_pair(p) is None
    assert np.asarray(load_rail_pairs(p)).size == 0
