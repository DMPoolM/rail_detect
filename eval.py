from __future__ import annotations

import argparse

import torch

from train import evaluate, load_cfg, make_loaders
from models.ufld_rail import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    metrics = evaluate(model, val_loader, cfg["anchors"]["num_grid"], device)
    print(metrics)


if __name__ == "__main__":
    main()
