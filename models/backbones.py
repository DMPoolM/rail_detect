from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


@dataclass
class BackboneInfo:
    channels: list[int]  # channels of the 3 feature maps returned by forward()
    strides: list[int]


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.info = BackboneInfo(channels=[128, 256, 512], strides=[8, 16, 32])

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2, c3, c4


class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        feats = m.features
        # Stride breakpoints for mobilenet_v3_small features:
        #   idx 0: stride 2
        #   idx 1: stride 4
        #   idx 2-3: stride 8  (c2, 24 ch)
        #   idx 4-8: stride 16 (c3, 48 ch)
        #   idx 9-12: stride 32 (c4, 576 ch at last)
        self.stage_low = feats[:2]      # -> stride 4
        self.stage_c2 = feats[2:4]      # -> stride 8, 24 ch
        self.stage_c3 = feats[4:9]      # -> stride 16, 48 ch
        self.stage_c4 = feats[9:]       # -> stride 32, 576 ch
        self.info = BackboneInfo(channels=[24, 48, 576], strides=[8, 16, 32])

    def forward(self, x: torch.Tensor):
        x = self.stage_low(x)
        c2 = self.stage_c2(x)
        c3 = self.stage_c3(c2)
        c4 = self.stage_c4(c3)
        return c2, c3, c4


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        return ResNet18Backbone(pretrained=pretrained)
    if name == "mobilenet_v3_small":
        return MobileNetV3SmallBackbone(pretrained=pretrained)
    raise ValueError(f"unknown backbone: {name}")
