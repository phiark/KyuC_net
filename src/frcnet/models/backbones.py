from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, image_batch):
        return self.backbone(image_batch)


def build_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    if backbone_name != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = ResNet18FeatureExtractor()
    return backbone, backbone.feature_dim

