"""RailSem19 JSON parser and ego-rail-pair selection.

Format reference (verified against the dataset authors' own visualization tool,
``example_vis.py`` shipped with RailSem19): a JSON file is

    {
      "frame": "rs19_val_00000",     # id of the matching .jpg
      "imgWidth": 1920, "imgHeight": 1080,
      "objects": [
        {"label": "rail", "polyline-pair": [[[x, y], ...], [[x, y], ...]]},
        {"label": "rail", "polyline":       [[x, y], ...]},
        {"label": "guard-rail", ...}, {"label": "rail-occluder", ...}
      ]
    }

Index 0 of ``polyline-pair`` is the left rail, index 1 the right rail. We ignore
``guard-rail`` and ``rail-occluder``: different geometry, not the running track.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
    if pts.ndim == 1 and pts.size % 2 == 0 and pts.size >= 4:
        return pts.reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def load_json(json_path: str | os.PathLike) -> dict:
    """Read a RailSem19 annotation file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def read_meta(json_path: str | os.PathLike) -> dict[str, object]:
    """Return ``{"frame": str, "img_width": int, "img_height": int}``.

    Values are best-effort: older or hand-trimmed exports may omit them, in which
    case ``frame`` falls back to the file stem and the sizes to ``None``.
    """
    data = load_json(json_path)
    w = data.get("imgWidth")
    h = data.get("imgHeight")
    return {
        "frame": str(data.get("frame") or Path(json_path).stem),
        "img_width": int(w) if w else None,
        "img_height": int(h) if h else None,
    }


def _polylines_from(data: dict, pairs_only: bool) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for obj in data.get("objects", []):
        if obj.get("label") != "rail":
            continue
        if "polyline-pair" in obj:
            for poly in obj["polyline-pair"][:2]:
                arr = _poly_to_np(poly)
                if len(arr) >= 2:
                    out.append(arr)
        elif not pairs_only and "polyline" in obj:
            arr = _poly_to_np(obj["polyline"])
            if len(arr) >= 2:
                out.append(arr)
    return out


def load_rail_polylines(json_path: str | os.PathLike) -> list[np.ndarray]:
    """Flat list of rail polylines (each (N, 2)).

    A ``polyline-pair`` object contributes 2 entries, a ``polyline`` object 1.
    """
    return _polylines_from(load_json(json_path), pairs_only=False)


def load_rail_pairs(json_path: str | os.PathLike) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (left, right) pairs from ``polyline-pair`` objects. Empty if none."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for obj in load_json(json_path).get("objects", []):
        if obj.get("label") != "rail" or "polyline-pair" not in obj:
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


def is_valid_pair(pair: RailPair) -> bool:
    """Left rail must lie left of the right rail, and both need >= 2 points."""
    return (
        len(pair.left) >= 2
        and len(pair.right) >= 2
        and _bottom_x(pair.left) < _bottom_x(pair.right)
    )


def select_ego_rail_pair(
    json_path: str | os.PathLike,
    img_w: int | None = None,
    img_h: int | None = None,
) -> RailPair | None:
    """Heuristically select the ego-vehicle rail pair.

    ``img_w``/``img_h`` default to the ``imgWidth``/``imgHeight`` recorded in the
    JSON, so callers that only have the annotation file can still use this.

    Preferred: if the JSON has explicit ``polyline-pair`` objects, pick the pair
    whose two bottom ends straddle the image centre most tightly.

    Fallback: take every rail polyline that reaches near the image bottom and pair
    the two whose bottom x's sit closest to the centre, one on each side.
    """
    data = load_json(json_path)
    if img_w is None:
        img_w = data.get("imgWidth")
    if img_h is None:
        img_h = data.get("imgHeight")
    if not img_w or not img_h:
        raise ValueError(
            f"{json_path}: image size unknown; pass img_w/img_h explicitly "
            "(the JSON has no imgWidth/imgHeight)"
        )
    cx = float(img_w) / 2.0
    bottom_thresh = float(img_h) * 0.85

    # --- preferred: explicit polyline-pair objects -------------------------
    # A pair is accepted if at least one of its rails reaches down to the bottom of
    # the frame; a pair where both stop halfway up is some other track, not the ego
    # one. (One rail may legitimately end early -- occluded by the cab, a crossing,
    # or the annotation simply stopping -- and its rows then read as "no rail".)
    best: RailPair | None = None
    best_dist = float("inf")
    for obj in data.get("objects", []):
        if obj.get("label") != "rail" or "polyline-pair" not in obj:
            continue
        left = _poly_to_np(obj["polyline-pair"][0])
        right = _poly_to_np(obj["polyline-pair"][1])
        if len(left) < 2 or len(right) < 2:
            continue
        if _bottom_y(left) < bottom_thresh and _bottom_y(right) < bottom_thresh:
            continue
        if _bottom_x(left) > _bottom_x(right):
            left, right = right, left
        d = abs(0.5 * (_bottom_x(left) + _bottom_x(right)) - cx)
        if d < best_dist:
            best_dist = d
            best = RailPair(left=left, right=right)
    if best is not None:
        return best

    # --- fallback: pair loose polylines by proximity to the centre ----------
    polys = [p for p in _polylines_from(data, pairs_only=False) if _bottom_y(p) >= bottom_thresh]
    left_cands = sorted((p for p in polys if _bottom_x(p) < cx), key=lambda p: cx - _bottom_x(p))
    right_cands = sorted((p for p in polys if _bottom_x(p) >= cx), key=lambda p: _bottom_x(p) - cx)
    if not left_cands or not right_cands:
        return None
    return RailPair(left=left_cands[0], right=right_cands[0])


def resolve_image_path(json_path: str | os.PathLike, img_dir: str | os.PathLike) -> Path:
    """Locate the .jpg for an annotation file, preferring the JSON's ``frame`` field.

    The dataset authors' own tooling keys images off ``data["frame"]`` rather than the
    JSON filename, so honour that first and fall back to the stem.
    """
    img_dir = Path(img_dir)
    try:
        frame = read_meta(json_path)["frame"]
    except (OSError, json.JSONDecodeError):
        frame = Path(json_path).stem
    for candidate in (img_dir / f"{frame}.jpg", img_dir / f"{Path(json_path).stem}.jpg"):
        if candidate.exists():
            return candidate
    return img_dir / f"{frame}.jpg"


def _dump_samples(root: str, json_subdir: str, img_subdir: str, out_dir: str, n: int) -> None:
    import cv2

    json_dir = Path(root) / json_subdir
    img_dir = Path(root) / img_subdir
    if not json_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {json_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    os.makedirs(out_dir, exist_ok=True)

    json_files: Sequence[Path] = sorted(json_dir.glob("*.json"))
    random.seed(0)
    json_files = list(json_files)
    random.shuffle(json_files)

    picked = 0
    skipped = 0
    for jp in json_files:
        if picked >= n:
            break
        img_path = resolve_image_path(jp, img_dir)
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        pair = select_ego_rail_pair(jp, w, h)
        if pair is None:
            skipped += 1
            continue
        for poly, color in ((pair.left, (0, 0, 255)), (pair.right, (255, 0, 0))):
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=3)
        cv2.imwrite(str(Path(out_dir) / f"{jp.stem}.jpg"), img)
        picked += 1
    print(f"Dumped {picked} samples to {out_dir} (skipped {skipped})")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/railsem19")
    parser.add_argument("--img-subdir", default="jpgs/rs19_val")
    parser.add_argument("--json-subdir", default="jsons/rs19_val")
    parser.add_argument("--out-dir", default="debug/ego_pair_samples")
    parser.add_argument("--dump-samples", type=int, default=20)
    args = parser.parse_args()
    _dump_samples(args.root, args.json_subdir, args.img_subdir, args.out_dir, args.dump_samples)


if __name__ == "__main__":
    _main()
