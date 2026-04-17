from __future__ import annotations

import torch
import torch.nn as nn

from frcnet.models.backbones import build_backbone
from frcnet.models.content_head import ContentHead
from frcnet.models.output_contracts import ModelOutput
from frcnet.models.resolution_head import ResolutionHead


class FRCNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        resolution_temperature: float = 1.0,
        content_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if resolution_temperature <= 0 or content_temperature <= 0:
            raise ValueError("Temperatures must be positive.")

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.resolution_temperature = float(resolution_temperature)
        self.content_temperature = float(content_temperature)

        self.backbone, feature_dim = build_backbone(backbone_name)
        self.resolution_head = ResolutionHead(feature_dim)
        self.content_head = ContentHead(feature_dim, num_classes)

    def forward(self, image_batch: torch.Tensor) -> ModelOutput:
        backbone_feature = self.backbone(image_batch)
        resolution_logit = self.resolution_head(backbone_feature)
        resolution_ratio = torch.sigmoid(resolution_logit / self.resolution_temperature)

        content_logits = self.content_head(backbone_feature)
        content_distribution = torch.softmax(content_logits / self.content_temperature, dim=-1)

        class_mass = resolution_ratio.unsqueeze(-1) * content_distribution
        unknown_mass = 1.0 - resolution_ratio

        return ModelOutput(
            backbone_feature=backbone_feature,
            resolution_logit=resolution_logit,
            resolution_ratio=resolution_ratio,
            content_logits=content_logits,
            content_distribution=content_distribution,
            class_mass=class_mass,
            unknown_mass=unknown_mass,
        )

