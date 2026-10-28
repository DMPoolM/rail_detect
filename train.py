from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.railsem19 import RailSem19Dataset, split_ids
from losses import RowAnchorLoss
from models.ufld_rail import build_model
from utils.config import load_cfg  # re-exported: `from train import load_cfg` still works
from utils.metrics import evaluate, format_metrics

__all__ = ["load_cfg", "make_loaders", "evaluate", "cosine_lr", "set_seed", "main"]


def set_seed(seed: int) -> None:
    """Seed everything we can reach. MPS/CUDA kernels stay non-deterministic, but
    data order, augmentation and init become reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    a = cfg["anchors"]
    inp = cfg["input"]
    t = cfg["train"]

    all_ids = RailSem19Dataset.list_ids(d["root"], d["img_subdir"], d["json_subdir"])
    if not all_ids:
        raise RuntimeError(
            f"no (image, annotation) pairs found under {d['root']} "
            f"(img_subdir={d['img_subdir']}, json_subdir={d['json_subdir']})"
        )
    train_ids, val_ids = split_ids(all_ids, d["val_ratio"], d["seed"])
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"bad split: {len(all_ids)} ids -> {len(train_ids)} train / {len(val_ids)} val; "
            "need at least 2 samples or a larger data.val_ratio"
        )

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

    # pin_memory only helps (and is only supported) for CUDA transfers
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=t["num_workers"],
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
        pin_memory=pin,
    )
    return train_loader, val_loader


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def _open_log(path: Path) -> CsvLogger:
    return CsvLogger(path)


class CsvLogger:
    """One-row-per-epoch CSV log that tolerates new metric columns appearing later.

    Existing rows are read back on construction, so resuming a run appends to the
    history instead of truncating it.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.columns: list[str] = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.columns = list(reader.fieldnames or [])
                self.rows = [dict(r) for r in reader]

    def log(self, row: dict[str, Any]) -> None:
        for k in row:
            if k not in self.columns:
                self.columns.append(k)
        self.rows.append(dict(row))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in self.columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the row-anchor rail detector.")
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from (e.g. checkpoints/last.pt)")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["train"]["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")

    train_loader, val_loader = make_loaders(cfg)
    print(f"samples: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

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

    # AMP only pays off on CUDA; MPS fp16 support is still patchy, so gate it.
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"amp: {use_amp}")

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        if "scaler" in state and use_amp:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best = float(state.get("best", -1.0))
        print(f"resumed from {args.resume} @ epoch {start_epoch} (best={best:.4f})")

    log_path = ckpt_dir / "metrics.csv"
    writer = CsvLogger(log_path)

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = int(cfg["train"]["warmup_epochs"] * len(train_loader))
    step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        run_total = run_cls = run_seg = 0.0
        n_steps = 0
        lr = cfg["train"]["lr"]  # keeps the log line defined if the loader is empty
        for img, label, seg, seg_valid in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            seg = seg.to(device, non_blocking=True)
            seg_valid = seg_valid.to(device, non_blocking=True)

            lr = cosine_lr(step, total_steps, cfg["train"]["lr"], warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, aux = model(img)
                loss_dict = loss_fn(logits, label, aux, seg, seg_valid)
                loss = loss_dict["total"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            run_total += loss.item()
            run_cls += float(loss_dict["cls"])
            run_seg += float(loss_dict["seg"])
            n_steps += 1

            if step % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    cls=f"{float(loss_dict['cls']):.3f}",
                    seg=f"{float(loss_dict['seg']):.3f}",
                    lr=f"{lr:.2e}",
                )
            step += 1

        metrics = evaluate(
            model,
            val_loader,
            cfg["anchors"]["num_grid"],
            device,
            input_w=cfg["input"]["width"],
        )
        print(f"[epoch {epoch}] {format_metrics(metrics)}")

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "best": best,
            "metrics": metrics,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if metrics["row_acc"] > best:
            best = metrics["row_acc"]
            state["best"] = best
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  new best row_acc={best:.4f} -> {ckpt_dir / 'best.pt'}")

        writer.log(
            {
                "epoch": epoch,
                "lr": lr,
                "train_total": run_total / max(1, n_steps),
                "train_cls": run_cls / max(1, n_steps),
                "train_seg": run_seg / max(1, n_steps),
                **{k: metrics[k] for k in metrics},
            }
        )



if __name__ == "__main__":
    main()
