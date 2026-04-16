from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone


class AuxSegHead(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int], mid: int = 64) -> None:
        super().__init__()
        c2, c3, c4 = in_channels
        self.lat2 = nn.Conv2d(c2, mid, 1)
        self.lat3 = nn.Conv2d(c3, mid, 1)
        self.lat4 = nn.Conv2d(c4, mid, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, c2, c3, c4, out_size):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.fuse(p2)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class RowAnchorHead(nn.Module):
    def __init__(self, in_channels: int, num_rails: int, num_rows: int, num_grid: int) -> None:
        super().__init__()
        self.num_rails = num_rails
        self.num_rows = num_rows
        self.num_classes = num_grid + 1  # +1 for "no rail"
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 9 * 25, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_rails * num_rows * self.num_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.reduce(feat)
        x = F.adaptive_avg_pool2d(x, (9, 25))
        x = self.fc(x)
        return x.view(-1, self.num_rails, self.num_rows, self.num_classes)


class UFLDRail(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_rails: int,
        num_rows: int,
        num_grid: int,
        use_aux_seg: bool,
        pretrained: bool,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        ch = self.backbone.info.channels
        self.head = RowAnchorHead(
            in_channels=ch[2],
            num_rails=num_rails,
            num_rows=num_rows,
            num_grid=num_grid,
        )
        self.use_aux_seg = use_aux_seg
        self.aux = AuxSegHead(in_channels=tuple(ch)) if use_aux_seg else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c2, c3, c4 = self.backbone(x)
        row_logits = self.head(c4)
        aux = None
        if self.use_aux_seg and self.aux is not None and self.training:
            aux = self.aux(c2, c3, c4, out_size=x.shape[-2:])
        return row_logits, aux


def build_model(cfg) -> UFLDRail:
    m = cfg["model"]
    a = cfg["anchors"]
    return UFLDRail(
        backbone=m["backbone"],
        num_rails=m["num_rails"],
        num_rows=a["num_rows"],
        num_grid=a["num_grid"],
        use_aux_seg=m.get("use_aux_seg", True),
        pretrained=m.get("pretrained", True),
    )
