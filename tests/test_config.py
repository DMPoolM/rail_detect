"""Config loading: partial overrides merge, bad values fail early with clear messages."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.config import DEFAULTS, ConfigError, load_cfg, validate_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_cfg(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_shipped_config_loads_and_matches_defaults():
    cfg = load_cfg(REPO_ROOT / "configs" / "default.yaml")
    assert cfg["model"]["backbone"] == "resnet18"
    assert cfg["anchors"]["num_rows"] == 56
    assert cfg["input"] == {"width": 800, "height": 288}


def test_partial_override_keeps_the_rest_of_the_defaults(tmp_path):
    cfg = load_cfg(_write_cfg(tmp_path, {"input": {"width": 640}}))
    assert cfg["input"]["width"] == 640
    assert cfg["input"]["height"] == DEFAULTS["input"]["height"]  # untouched
    assert cfg["loss"]["lambda_cls"] == DEFAULTS["loss"]["lambda_cls"]


def test_empty_file_is_all_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_cfg(p)["train"]["epochs"] == DEFAULTS["train"]["epochs"]


@pytest.mark.parametrize(
    "override, match",
    [
        ({"model": {"backbone": "vgg16"}}, "unknown model.backbone"),
        ({"model": {"num_rails": 4}}, "only num_rails=2"),
        ({"anchors": {"row_start": 0.9, "row_end": 0.5}}, "row_start must be smaller"),
        ({"anchors": {"num_grid": 1}}, "num_grid"),
        ({"infer": {"conf": 0.0}}, "infer.conf"),
        ({"infer": {"ema": 1.0}}, "infer.ema"),
        ({"train": {"batch_size": 0}}, "batch_size"),
        ({"train": {"epochs": 2, "warmup_epochs": 5}}, "warmup_epochs"),
        ({"loss": {"lambda_cls": -1}}, "lambda_cls"),
        ({"data": {"rail_class_ids": []}}, "rail_class_ids"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, override, match):
    with pytest.raises(ConfigError, match=match):
        load_cfg(_write_cfg(tmp_path, override))


def test_missing_file_is_reported_as_config_error(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_cfg(tmp_path / "nope.yaml")


def test_non_mapping_yaml_is_rejected(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_cfg(p)


def test_width_not_divisible_by_32_warns_but_loads(tmp_path):
    with pytest.warns(UserWarning, match="multiple of 32"):
        cfg = load_cfg(_write_cfg(tmp_path, {"input": {"width": 641}}))
    assert cfg["input"]["width"] == 641


def test_validate_cfg_accepts_a_hand_built_dict():
    validate_cfg(DEFAULTS)  # should not raise
