"""RailSem19 JSON parser and ego-rail-pair selection.

RailSem19 annotates rails as objects with label == "rail". Each such object carries
either a "polyline-pair" (left+right polylines of one track) or a "polyline" (single
rail). We ignore "guard-rail" and "rail-occluder" which use different geometries and
are not part of the running-track pair.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RailPair:
    """A pair of (left, right) rail polylines, each as (N, 2) array of (x, y)."""
    left: np.ndarray
    right: np.ndarray


def _poly_to_np(poly) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts
    if pts.ndim == 1 and pts.size % 2 == 0:
        return pts.reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def load_rail_polylines(json_path: str | os.PathLike) -> List[np.ndarray]:
    """Return a flat list of rail polylines (each (N, 2)) from label=='rail' objects.
    A polyline-pair object contributes 2 entries, a polyline object contributes 1.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[np.ndarray] = []
    for obj in data.get("objects", []):
        if obj.get("label") != "rail":
            continue
        if "polyline-pair" in obj:
            left = _poly_to_np(obj["polyline-pair"][0])
            right = _poly_to_np(obj["polyline-pair"][1])
            if len(left) >= 2:
                out.append(left)
            if len(right) >= 2:
                out.append(right)
        elif "polyline" in obj:
            p = _poly_to_np(obj["polyline"])
            if len(p) >= 2:
                out.append(p)
    return out


def load_rail_pairs(json_path: str | os.PathLike) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (left, right) rail polyline pairs from label=='rail' objects that
    carry 'polyline-pair'. Empty if none are pair-encoded.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for obj in data.get("objects", []):
        if obj.get("label") != "rail":
            continue
        if "polyline-pair" not in obj:
            continue
        left = _poly_to_np(obj["polyline-pair"][0])
        right = _poly_to_np(obj["polyline-pair"][1])
        if len(left) >= 2 and len(right) >= 2:
            pairs.append((left, right))
    return pairs


def _bottom_x(poly: np.ndarray) -> float:
    """X value at the bottommost point (largest y) of a polyline."""
    i = int(np.argmax(poly[:, 1]))
    return float(poly[i, 0])


def _bottom_y(poly: np.ndarray) -> float:
    return float(np.max(poly[:, 1]))


def select_ego_rail_pair(
    json_path: str | os.PathLike,
    img_w: int,
    img_h: int,
) -> Optional[RailPair]:
    """Heuristically select the ego-vehicle rail pair.

    Preferred: if the JSON has explicit polyline-pair objects, pick the pair whose
    mid-point of the bottom ends is closest to the image bottom center.

    Fallback: take all rail polylines that reach near the image bottom, and greedily
    pair the two whose bottom x straddle the image center x and sit closest to it.
    """
    cx = img_w / 2.0
    bottom_thresh = img_h * 0.85

    pairs = load_rail_pairs(json_path)
    if pairs:
        best = None
        best_dist = float("inf")
        for left, right in pairs:
            if _bottom_y(left) < bottom_thresh and _bottom_y(right) < bottom_thresh:
                continue
            mid = 0.5 * (_bottom_x(left) + _bottom_x(right))
            d = abs(mid - cx)
            if d < best_dist:
                best_dist = d
                l, r = (left, right) if _bottom_x(left) < _bottom_x(right) else (right, left)
                best = RailPair(left=l, right=r)
        if best is not None:
            return best

    polys = load_rail_polylines(json_path)
    polys = [p for p in polys if _bottom_y(p) >= bottom_thresh]
    if len(polys) < 2:
        return None

    left_cands = sorted(
        [p for p in polys if _bottom_x(p) < cx],
        key=lambda p: cx - _bottom_x(p),
    )
    right_cands = sorted(
        [p for p in polys if _bottom_x(p) >= cx],
        key=lambda p: _bottom_x(p) - cx,
    )
    if not left_cands or not right_cands:
        return None
    return RailPair(left=left_cands[0], right=right_cands[0])


def _dump_samples(root: str, json_subdir: str, img_subdir: str, out_dir: str, n: int) -> None:
    import cv2

    json_dir = Path(root) / json_subdir
    img_dir = Path(root) / img_subdir
    os.makedirs(out_dir, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    random.seed(0)
    random.shuffle(json_files)

    picked = 0
    for jp in json_files:
        if picked >= n:
            break
        img_path = img_dir / (jp.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        pair = select_ego_rail_pair(jp, w, h)
        if pair is None:
            continue
        for poly, color in ((pair.left, (0, 0, 255)), (pair.right, (255, 0, 0))):
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=3)
        cv2.imwrite(str(Path(out_dir) / f"{jp.stem}.jpg"), img)
        picked += 1
    print(f"Dumped {picked} samples to {out_dir}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/railsem19")
    parser.add_argument("--img-subdir", default="rs19_val/jpgs/rs19_val")
    parser.add_argument("--json-subdir", default="rs19_val/jsons/rs19_val")
    parser.add_argument("--out-dir", default="debug/ego_pair_samples")
    parser.add_argument("--dump-samples", type=int, default=20)
    args = parser.parse_args()
    _dump_samples(args.root, args.json_subdir, args.img_subdir, args.out_dir, args.dump_samples)


if __name__ == "__main__":
    _main()
