from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.anchor import build_anchor_rows, interp_polyline_at_ys, x_to_grid
from utils.railsem_parser import select_ego_rail_pair

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RailSem19Dataset(Dataset):
    """Row-anchor dataset for ego rail pair from RailSem19.

    ``__getitem__`` returns ``(image, label, seg, seg_valid)``:

    * ``image``     — (3, H, W) float32, ImageNet-normalised
    * ``label``     — (K=2, num_rows) int64 in ``[0, num_grid]``; ``num_grid`` means
      "no rail on this anchor row"
    * ``seg``       — (H, W) float32 binary rail mask for the auxiliary head
    * ``seg_valid`` — scalar float32, 0.0 when the mask file is missing

    ``seg_valid`` exists because a missing mask used to be replaced with an all-zero
    mask. That teaches the auxiliary head "this scene has no rails", which fights the
    real seg loss on every other sample — so the loss now ignores invalid samples
    instead (see :class:`losses.RowAnchorLoss`).
    """

    def __init__(
        self,
        root: str,
        img_subdir: str,
        json_subdir: str,
        seg_subdir: str,
        rail_class_ids: list[int],
        ids: list[str],
        input_w: int,
        input_h: int,
        num_rows: int,
        num_grid: int,
        row_start: float,
        row_end: float,
        train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / img_subdir
        self.json_dir = self.root / json_subdir
        self.seg_dir = self.root / seg_subdir
        self.rail_class_ids = set(rail_class_ids)
        self.ids = ids
        self.input_w = input_w
        self.input_h = input_h
        self.num_rows = num_rows
        self.num_grid = num_grid
        self.row_start = row_start
        self.row_end = row_end
        self.train = train

        self.anchor_ys_input = build_anchor_rows(num_rows, row_start, row_end, input_h)

    @staticmethod
    def list_ids(root: str, img_subdir: str, json_subdir: str | None = None) -> list[str]:
        """Frame ids that have an image, and (when given) a matching annotation file.

        Filtering on the annotation here turns a dataset with missing/mis-named JSON
        files into a readable "0 samples" error at startup instead of a
        ``FileNotFoundError`` thrown from inside a worker mid-training.
        """
        d = Path(root) / img_subdir
        if not d.is_dir():
            raise FileNotFoundError(f"image directory not found: {d}")
        ids = sorted(p.stem for p in d.glob("*.jpg"))
        if json_subdir is not None:
            jd = Path(root) / json_subdir
            if not jd.is_dir():
                raise FileNotFoundError(f"annotation directory not found: {jd}")
            ids = [i for i in ids if (jd / f"{i}.json").exists()]
        return ids

    def __len__(self) -> int:
        return len(self.ids)

    def _load_seg(self, stem: str) -> tuple[np.ndarray, bool]:
        """Return ``(rail_mask, is_valid)``.

        ``is_valid`` is False when the dense label map is absent or unreadable; the
        returned mask is then all-zero and the aux-seg loss will ignore this sample.
        """
        p = self.seg_dir / f"{stem}.png"
        empty = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        if not p.exists():
            return empty, False
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            return empty, False
        rail = np.isin(m, list(self.rail_class_ids)).astype(np.uint8)
        seg = cv2.resize(rail, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)
        return seg, True

    def _build_row_label(
        self,
        left: np.ndarray,
        right: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        sx = self.input_w / orig_w
        sy = self.input_h / orig_h
        left_s = left.copy()
        right_s = right.copy()
        left_s[:, 0] *= sx
        left_s[:, 1] *= sy
        right_s[:, 0] *= sx
        right_s[:, 1] *= sy

        label = np.full((2, self.num_rows), self.num_grid, dtype=np.int64)
        for k, poly in enumerate((left_s, right_s)):
            xs = interp_polyline_at_ys(poly, self.anchor_ys_input)
            valid = ~np.isnan(xs)
            if valid.any():
                g = x_to_grid(xs[valid], self.input_w, self.num_grid)
                label[k, valid] = g
        return label

    def _augment(
        self,
        img: np.ndarray,
        seg: np.ndarray,
        label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            seg = seg[:, ::-1].copy()
            flipped = np.full_like(label, self.num_grid)
            valid = label < self.num_grid
            flipped[valid] = self.num_grid - 1 - label[valid]
            label = flipped[[1, 0]]

        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 2] *= random.uniform(0.7, 1.3)
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, seg, label

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_path = self.img_dir / f"{stem}.jpg"
        json_path = self.json_dir / f"{stem}.json"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"no annotation for image {img_path.name}: expected {json_path}"
            )
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        pair = select_ego_rail_pair(json_path, orig_w, orig_h)
        if pair is None:
            label = np.full((2, self.num_rows), self.num_grid, dtype=np.int64)
        else:
            label = self._build_row_label(pair.left, pair.right, orig_w, orig_h)

        seg, seg_valid = self._load_seg(stem)

        if self.train:
            img_rs, seg, label = self._augment(img_rs, seg, label)

        img_f = img_rs.astype(np.float32) / 255.0
        img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1)))
        label_t = torch.from_numpy(label)
        seg_t = torch.from_numpy(np.ascontiguousarray(seg.astype(np.float32)))
        seg_valid_t = torch.tensor(1.0 if seg_valid else 0.0, dtype=torch.float32)

        return img_t, label_t, seg_t, seg_valid_t


def split_ids(ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    return ids[n_val:], ids[:n_val]
