from __future__ import annotations

import torch.nn as nn


class ContentHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, num_classes)

    def forward(self, backbone_feature):
        return self.projection(backbone_feature)

