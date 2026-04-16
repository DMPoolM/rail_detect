from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.railsem19 import RailSem19Dataset, split_ids
from losses import RowAnchorLoss
from models.ufld_rail import build_model


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loaders(cfg: dict):
    d = cfg["data"]
    a = cfg["anchors"]
    inp = cfg["input"]
    t = cfg["train"]

    all_ids = RailSem19Dataset.list_ids(d["root"], d["img_subdir"])
    train_ids, val_ids = split_ids(all_ids, d["val_ratio"], d["seed"])

    common = dict(
        root=d["root"],
        img_subdir=d["img_subdir"],
        json_subdir=d["json_subdir"],
        seg_subdir=d["seg_subdir"],
        rail_class_ids=d["rail_class_ids"],
        input_w=inp["width"],
        input_h=inp["height"],
        num_rows=a["num_rows"],
        num_grid=a["num_grid"],
        row_start=a["row_start"],
        row_end=a["row_end"],
    )
    train_ds = RailSem19Dataset(ids=train_ids, train=True, **common)
    val_ds = RailSem19Dataset(ids=val_ids, train=False, **common)

    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=t["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, num_grid, device) -> dict:
    model.eval()
    total_rows = 0
    correct_rows = 0
    total_present = 0
    correct_present = 0
    for img, label, _seg in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits, _ = model(img)
        pred = logits.argmax(dim=-1)

        present_gt = label < num_grid
        present_pred = pred < num_grid
        total_present += present_gt.numel()
        correct_present += (present_gt == present_pred).sum().item()

        mask = present_gt & present_pred
        if mask.any():
            diff = (pred[mask].long() - label[mask].long()).abs()
            correct_rows += (diff <= 2).sum().item()
            total_rows += mask.sum().item()

    row_acc = correct_rows / max(1, total_rows)
    present_acc = correct_present / max(1, total_present)
    return {"row_acc": row_acc, "present_acc": present_acc}


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(cfg)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.2f}M")

    loss_fn = RowAnchorLoss(
        num_grid=cfg["anchors"]["num_grid"],
        lambda_cls=cfg["loss"]["lambda_cls"],
        lambda_sim=cfg["loss"]["lambda_sim"],
        lambda_shp=cfg["loss"]["lambda_shp"],
        lambda_seg=cfg["loss"]["lambda_seg"],
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = cfg["train"]["warmup_epochs"] * len(train_loader)
    step = 0
    best = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for img, label, seg in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            seg = seg.to(device, non_blocking=True)

            lr = cosine_lr(step, total_steps, cfg["train"]["lr"], warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            logits, aux = model(img)
            loss_dict = loss_fn(logits, label, aux, seg)
            loss = loss_dict["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    cls=f"{loss_dict['cls'].item():.3f}",
                    seg=f"{loss_dict['seg'].item():.3f}",
                    lr=f"{lr:.2e}",
                )
            step += 1

        metrics = evaluate(model, val_loader, cfg["anchors"]["num_grid"], device)
        print(f"[epoch {epoch}] row_acc={metrics['row_acc']:.4f} present_acc={metrics['present_acc']:.4f}")

        state = {
            "model": model.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if metrics["row_acc"] > best:
            best = metrics["row_acc"]
            torch.save(state, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
