from __future__ import annotations

import argparse

import torch

from models.ufld_rail import build_model
from train import make_loaders
from utils.config import load_cfg
from utils.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the validation split.")
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if args.device != "auto":
        device = torch.device(args.device)

    _, val_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"checkpoint: {args.ckpt} (epoch {state.get('epoch', '?')}, device {device})")
    metrics = evaluate(
        model,
        val_loader,
        cfg["anchors"]["num_grid"],
        device,
        input_w=cfg["input"]["width"],
    )

    if args.json:
        import json

        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    for k in sorted(metrics):
        print(f"{k:22s} {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
