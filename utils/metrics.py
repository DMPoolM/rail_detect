"""Validation metrics for row-anchor rail detection.

Why this is not just ``(pred == gt).mean()``: most anchor rows are "no rail", so a
model that predicts "no rail" everywhere would score high on a naive row accuracy.
The metrics below therefore separate two questions:

* **presence** — is there a rail at this row at all? (precision / recall / F1)
* **localisation** — given that both agree a rail is there, is the x prediction
  within ``tol`` grid cells? (``row_acc_detected``, the metric the original README
  quoted)

``row_acc`` is the stricter, more honest number: every ground-truth rail row counts,
so predicting "no rail" on a real row is an error rather than being skipped.
"""

from __future__ import annotations

import torch


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    num_grid: int,
    device,
    tol: float = 2.0,
    input_w: int | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run the model over ``loader`` and return a flat dict of metrics.

    ``tol`` is the localisation tolerance in grid cells (±2 → ±16 px at 800 px wide
    with 100 cells). If ``input_w`` is given, ``row_mae_px`` is reported as well.
    """
    was_training = model.training
    model.eval()

    stats: list[dict[str, float]] = []
    total_rows = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        img, label = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        logits, _ = model(img)
        pred = logits.argmax(dim=-1)

        gt_present = label < num_grid
        pr_present = pred < num_grid
        both = gt_present & pr_present
        diff = (pred - label).abs()
        hit = both & (diff <= tol)

        if not stats:
            stats = [{"tp": 0, "gt": 0, "pd": 0, "hit": 0, "diff": 0.0} for _ in range(pred.shape[1])]
        total_rows += pred.numel()
        for k in range(pred.shape[1]):
            s = stats[k]
            both_k = both[:, k]
            s["tp"] += int(both_k.sum())
            s["gt"] += int(gt_present[:, k].sum())
            s["pd"] += int(pr_present[:, k].sum())
            s["hit"] += int(hit[:, k].sum())
            s["diff"] += float(diff[:, k][both_k].float().sum())

    if was_training:
        model.train()

    if not stats:
        return {"presence_precision": 0.0, "presence_recall": 0.0, "presence_f1": 0.0,
                "row_acc": 0.0, "row_acc_detected": 0.0, "row_mae_cells": 0.0,
                "pred_present_rate": 0.0, "n_rows": 0.0}

    tp = sum(s["tp"] for s in stats)
    gt = sum(s["gt"] for s in stats)
    pd = sum(s["pd"] for s in stats)
    hit = sum(s["hit"] for s in stats)
    diff = sum(s["diff"] for s in stats)

    precision = _safe_div(tp, pd)
    recall = _safe_div(tp, gt)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mae_cells = _safe_div(diff, tp)

    out: dict[str, float] = {
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "row_acc": _safe_div(hit, gt),
        "row_acc_detected": _safe_div(hit, tp),
        "row_mae_cells": mae_cells,
        "pred_present_rate": _safe_div(pd, total_rows),
        "n_rows": float(total_rows),
    }
    if input_w:
        out["row_mae_px"] = mae_cells * float(input_w) / float(num_grid)

    names = ("left", "right")
    for k, s in enumerate(stats):
        name = names[k] if k < len(names) else f"rail{k}"
        out[f"{name}_precision"] = _safe_div(s["tp"], s["pd"])
        out[f"{name}_recall"] = _safe_div(s["tp"], s["gt"])
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    """One-line summary for logs and the CLI."""
    return (
        f"row_acc={metrics['row_acc']:.4f} "
        f"(detected={metrics['row_acc_detected']:.4f}, ±{metrics['row_mae_cells']:.2f} cells) "
        f"P={metrics['presence_precision']:.4f} R={metrics['presence_recall']:.4f} "
        f"F1={metrics['presence_f1']:.4f}"
    )
