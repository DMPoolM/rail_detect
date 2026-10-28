"""Shared fixtures: a miniature, self-contained RailSem19-shaped dataset.

Nothing here needs the real dataset (~1.6 GB and a registration wall), so the whole
suite runs in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_sample(
    root: Path,
    stem: str,
    w: int = 64,
    h: int = 32,
    *,
    with_json: bool = True,
    with_seg: bool = True,
    rail_ids=(12,),
) -> None:
    """Create one image / annotation / label-map triple under ``root``."""
    img_dir = root / "jpgs"
    json_dir = root / "jsons"
    seg_dir = root / "uint8"
    for d in (img_dir, json_dir, seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)

    if with_json:
        pair = {
            "label": "rail",
            "polyline-pair": [
                [[int(w * 0.20), 0], [int(w * 0.35), h]],
                [[int(w * 0.80), 0], [int(w * 0.65), h]],
            ],
        }
        decoy = {
            "label": "rail",
            "polyline-pair": [
                [[int(w * 0.02), 0], [int(w * 0.02), h]],
                [[int(w * 0.06), 0], [int(w * 0.06), h]],
            ],
        }
        guard = {"label": "guard-rail", "polyline": [[0, 0], [w - 1, h - 1]]}
        data = {
            "frame": stem,
            "imgWidth": w,
            "imgHeight": h,
            "objects": [decoy, pair, guard],
        }
        with open(json_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    if with_seg:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, w // 3 : 2 * w // 3] = rail_ids[0]
        cv2.imwrite(str(seg_dir / f"{stem}.png"), mask)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Three samples: two complete, one missing its label map."""
    root = tmp_path / "railsem19"
    write_sample(root, "rs19_val_00000")
    write_sample(root, "rs19_val_00001")
    write_sample(root, "rs19_val_00002", with_seg=False)
    return root


@pytest.fixture
def dataset_kwargs(tiny_dataset: Path) -> dict:
    return dict(
        root=str(tiny_dataset),
        img_subdir="jpgs",
        json_subdir="jsons",
        seg_subdir="uint8",
        rail_class_ids=[12, 17, 18],
        input_w=64,
        input_h=32,
        num_rows=8,
        num_grid=20,
        row_start=0.42,
        row_end=1.0,
    )
