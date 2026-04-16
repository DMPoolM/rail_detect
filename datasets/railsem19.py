from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

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

    Label tensor shape: (K=2, num_rows). Each entry is an integer in [0, num_grid].
    The value `num_grid` means "no rail on this anchor row".
    """

    def __init__(
        self,
        root: str,
        img_subdir: str,
        json_subdir: str,
        seg_subdir: str,
        rail_class_ids: List[int],
        ids: List[str],
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
    def list_ids(root: str, img_subdir: str) -> List[str]:
        d = Path(root) / img_subdir
        return sorted(p.stem for p in d.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_seg(self, stem: str) -> np.ndarray:
        p = self.seg_dir / f"{stem}.png"
        if not p.exists():
            return np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        rail = np.isin(m, list(self.rail_class_ids)).astype(np.uint8)
        return cv2.resize(rail, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        pair = select_ego_rail_pair(json_path, orig_w, orig_h)
        if pair is None:
            label = np.full((2, self.num_rows), self.num_grid, dtype=np.int64)
        else:
            label = self._build_row_label(pair.left, pair.right, orig_w, orig_h)

        seg = self._load_seg(stem)

        if self.train:
            img_rs, seg, label = self._augment(img_rs, seg, label)

        img_f = img_rs.astype(np.float32) / 255.0
        img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1)))
        label_t = torch.from_numpy(label)
        seg_t = torch.from_numpy(seg.astype(np.float32))

        return img_t, label_t, seg_t


def split_ids(ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    return ids[n_val:], ids[:n_val]
