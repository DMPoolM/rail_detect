"""Configuration loading: YAML over in-code defaults, with up-front validation.

Two reasons this is a module of its own rather than a helper inside ``train.py``:

1. ``infer_video.py`` used to import ``load_cfg`` from ``train``, which dragged in
   ``torch.utils.data``, ``tqdm`` and the dataset stack just to read a YAML file.
2. A partial config file is a feature, but a *typo* in it must fail loudly and early
   rather than surfacing as a ``KeyError`` an hour into training.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

#: Canonical defaults. ``configs/default.yaml`` may override any subset of these.
DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "data/railsem19",
        "img_subdir": "jpgs/rs19_val",
        "json_subdir": "jsons/rs19_val",
        "seg_subdir": "uint8/rs19_val",
        "rail_class_ids": [12, 17, 18],
        "val_ratio": 0.1,
        "seed": 42,
    },
    "input": {"width": 800, "height": 288},
    "anchors": {"num_rows": 56, "num_grid": 100, "row_start": 0.42, "row_end": 1.0},
    "model": {
        "backbone": "resnet18",
        "pretrained": True,
        "num_rails": 2,
        "use_aux_seg": True,
    },
    "loss": {
        "lambda_cls": 1.0,
        "lambda_sim": 1.0,
        "lambda_shp": 0.0,
        "lambda_seg": 1.0,
    },
    "train": {
        "epochs": 50,
        "batch_size": 32,
        "num_workers": 2,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 1,
        "ckpt_dir": "checkpoints",
        "log_interval": 20,
        "amp": True,
        "seed": 42,
    },
    "infer": {"ema": 0.0, "conf": 0.3},
}

KNOWN_BACKBONES = ("resnet18", "mobilenet_v3_small")


class ConfigError(ValueError):
    """Raised when a config value is malformed or out of range."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"missing config key: {dotted}")
        node = node[part]
    return node


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_range(cfg, key, lo, hi, *, lo_open=False, hi_open=False) -> None:
    v = _get(cfg, key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ConfigError(f"{key} must be a number, got {type(v).__name__}")
    too_low = v <= lo if lo_open else v < lo
    too_high = v >= hi if hi_open else v > hi
    if too_low or too_high:
        left = "(" if lo_open else "["
        right = ")" if hi_open else "]"
        raise ConfigError(f"{key}={v} out of range {left}{lo}, {hi}{right}")


def validate_cfg(cfg: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` on the first problem found."""
    # --- data ---------------------------------------------------------------
    _check_range(cfg, "data.val_ratio", 0.0, 1.0, hi_open=True)
    ids = _get(cfg, "data.rail_class_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ConfigError("data.rail_class_ids must be a non-empty list of ints")
    for i in ids:
        if not _is_int(i) or not 0 <= i <= 255:
            raise ConfigError(f"data.rail_class_ids contains invalid id: {i!r}")

    # --- input --------------------------------------------------------------
    for key in ("input.width", "input.height"):
        v = _get(cfg, key)
        if not _is_int(v) or v <= 0:
            raise ConfigError(f"{key} must be a positive int, got {v!r}")
        if v % 32 != 0:
            warnings.warn(
                f"{key}={v} is not a multiple of 32; the stride-32 backbone will "
                "still run (adaptive pooling), but anchors may look slightly off.",
                stacklevel=2,
            )

    # --- anchors ------------------------------------------------------------
    for key in ("anchors.num_rows", "anchors.num_grid"):
        v = _get(cfg, key)
        if not _is_int(v) or v < 2:
            raise ConfigError(f"{key} must be an int >= 2, got {v!r}")
    _check_range(cfg, "anchors.row_start", 0.0, 1.0)
    _check_range(cfg, "anchors.row_end", 0.0, 1.0)
    if not _get(cfg, "anchors.row_start") < _get(cfg, "anchors.row_end"):
        raise ConfigError("anchors.row_start must be smaller than anchors.row_end")

    # --- model --------------------------------------------------------------
    backbone = _get(cfg, "model.backbone")
    if backbone not in KNOWN_BACKBONES:
        raise ConfigError(
            f"unknown model.backbone {backbone!r}; expected one of {list(KNOWN_BACKBONES)}"
        )
    if _get(cfg, "model.num_rails") != 2:
        raise ConfigError(
            "only num_rails=2 (ego left/right pair) is implemented; "
            f"got {_get(cfg, 'model.num_rails')!r}"
        )

    # --- loss ---------------------------------------------------------------
    for key in ("lambda_cls", "lambda_sim", "lambda_shp", "lambda_seg"):
        _check_range(cfg, f"loss.{key}", 0.0, float("inf"))

    # --- train --------------------------------------------------------------
    for key in ("train.epochs", "train.batch_size", "train.log_interval"):
        v = _get(cfg, key)
        if not _is_int(v) or v < 1:
            raise ConfigError(f"{key} must be an int >= 1, got {v!r}")
    v = _get(cfg, "train.num_workers")
    if not _is_int(v) or v < 0:
        raise ConfigError(f"train.num_workers must be an int >= 0, got {v!r}")
    _check_range(cfg, "train.lr", 0.0, float("inf"), lo_open=True)
    _check_range(cfg, "train.weight_decay", 0.0, float("inf"))
    warmup = _get(cfg, "train.warmup_epochs")
    if not isinstance(warmup, (int, float)) or warmup < 0:
        raise ConfigError(f"train.warmup_epochs must be >= 0, got {warmup!r}")
    if warmup > _get(cfg, "train.epochs"):
        raise ConfigError("train.warmup_epochs cannot exceed train.epochs")

    # --- infer --------------------------------------------------------------
    _check_range(cfg, "infer.conf", 0.0, 1.0, lo_open=True)
    _check_range(cfg, "infer.ema", 0.0, 1.0, hi_open=True)


def load_cfg(path: str | Path) -> dict[str, Any]:
    """Load ``path``, merge it over :data:`DEFAULTS`, validate, return a fresh dict."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(raw).__name__}")
    cfg = _deep_merge(DEFAULTS, raw)
    validate_cfg(cfg)
    return cfg
