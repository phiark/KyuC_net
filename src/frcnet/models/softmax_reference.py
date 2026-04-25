from __future__ import annotations

import torch
import torch.nn as nn

from frcnet.models.backbones import build_backbone


class SoftmaxReferenceModel(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_backbone(backbone_name)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(image_batch))
