from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from frcnet.models.backbones import build_backbone
from frcnet.models.content_head import ContentHead
from frcnet.models.output_contracts import ModelOutput
from frcnet.models.resolution_head import ResolutionHead


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


class SourceAdversaryHead(nn.Module):
    def __init__(self, feature_dim: int, num_source_domains: int, hidden_dim: int) -> None:
        super().__init__()
        if num_source_domains <= 1:
            raise ValueError("num_source_domains must be greater than one when source adversary is enabled.")
        if hidden_dim <= 0:
            raise ValueError("source_head_hidden_dim must be positive.")
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_source_domains),
        )

    def forward(self, feature_batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(feature_batch)


def gradient_reverse(input_tensor: torch.Tensor, lambda_value: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(input_tensor, float(lambda_value))


class FRCNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        resolution_temperature: float = 1.0,
        content_temperature: float = 1.0,
        source_adversary_enabled: bool = False,
        num_source_domains: int = 0,
        grl_lambda: float = 1.0,
        source_head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if resolution_temperature <= 0 or content_temperature <= 0:
            raise ValueError("Temperatures must be positive.")
        if grl_lambda < 0.0:
            raise ValueError("grl_lambda must be non-negative.")

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.resolution_temperature = float(resolution_temperature)
        self.content_temperature = float(content_temperature)
        self.source_adversary_enabled = bool(source_adversary_enabled)
        self.num_source_domains = int(num_source_domains)
        self.grl_lambda = float(grl_lambda)

        self.backbone, feature_dim = build_backbone(backbone_name)
        self.resolution_head = ResolutionHead(feature_dim)
        self.content_head = ContentHead(feature_dim, num_classes)
        self.source_head = (
            SourceAdversaryHead(feature_dim, self.num_source_domains, int(source_head_hidden_dim))
            if self.source_adversary_enabled
            else None
        )

    def forward(self, image_batch: torch.Tensor) -> ModelOutput:
        backbone_feature = self.backbone(image_batch)
        resolution_logit = self.resolution_head(backbone_feature)
        resolution_ratio = torch.sigmoid(resolution_logit / self.resolution_temperature)

        content_logits = self.content_head(backbone_feature)
        content_distribution = torch.softmax(content_logits / self.content_temperature, dim=-1)

        class_mass = resolution_ratio.unsqueeze(-1) * content_distribution
        unknown_mass = 1.0 - resolution_ratio
        source_logits = None
        if self.source_head is not None:
            source_logits = self.source_head(gradient_reverse(backbone_feature, self.grl_lambda))

        return ModelOutput(
            backbone_feature=backbone_feature,
            resolution_logit=resolution_logit,
            resolution_ratio=resolution_ratio,
            content_logits=content_logits,
            content_distribution=content_distribution,
            class_mass=class_mass,
            unknown_mass=unknown_mass,
            source_logits=source_logits,
        )
