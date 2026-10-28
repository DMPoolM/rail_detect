from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RowAnchorLoss(nn.Module):
    """UFLD-style losses for rail row-anchor prediction.

    Inputs:
      logits: (B, K, H, W+1) — raw logits, last class is "no rail"
      target: (B, K, H)     — integer class labels in [0, W]
      seg_logits: (B, 1, Hs, Ws) or None
      seg_gt:     (B, Hs, Ws) or None
      seg_valid:  (B,) or None — per-sample 1/0 mask; samples with a 0 are excluded
                  from the auxiliary seg loss instead of being trained on an
                  all-background mask.
    """

    def __init__(
        self,
        num_grid: int,
        lambda_cls: float = 1.0,
        lambda_sim: float = 1.0,
        lambda_shp: float = 0.0,
        lambda_seg: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_grid = num_grid
        self.lambda_cls = lambda_cls
        self.lambda_sim = lambda_sim
        self.lambda_shp = lambda_shp
        self.lambda_seg = lambda_seg

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        seg_logits: torch.Tensor | None = None,
        seg_gt: torch.Tensor | None = None,
        seg_valid: torch.Tensor | None = None,
    ) -> dict:
        B, K, H, C = logits.shape
        cls = F.cross_entropy(logits.reshape(-1, C), target.reshape(-1))

        grid_logits = logits[..., : self.num_grid]
        probs = F.softmax(grid_logits, dim=-1)
        if H >= 2:
            sim = F.l1_loss(probs[:, :, 1:], probs[:, :, :-1])
        else:
            sim = torch.zeros((), device=logits.device)

        if self.lambda_shp > 0 and H >= 3:
            centers = torch.arange(self.num_grid, device=logits.device, dtype=probs.dtype)
            exp_x = (probs * centers).sum(dim=-1)
            d2 = exp_x[:, :, 2:] - 2 * exp_x[:, :, 1:-1] + exp_x[:, :, :-2]
            shp = (d2.pow(2)).mean()
        else:
            shp = torch.zeros((), device=logits.device)

        if seg_logits is not None and seg_gt is not None:
            per_sample = F.binary_cross_entropy_with_logits(
                seg_logits.squeeze(1), seg_gt, reduction="none"
            ).flatten(1).mean(dim=1)
            if seg_valid is not None:
                w = seg_valid.to(per_sample.dtype).reshape(-1)
                seg = (per_sample * w).sum() / w.sum().clamp(min=1.0)
            else:
                seg = per_sample.mean()
        else:
            seg = torch.zeros((), device=logits.device)

        total = (
            self.lambda_cls * cls
            + self.lambda_sim * sim
            + self.lambda_shp * shp
            + self.lambda_seg * seg
        )
        return {"total": total, "cls": cls.detach(), "sim": sim.detach(), "shp": shp.detach(), "seg": seg.detach()}
