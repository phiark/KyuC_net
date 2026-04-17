from __future__ import annotations

import torch.nn as nn


class ResolutionHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, 1)

    def forward(self, backbone_feature):
        return self.projection(backbone_feature).squeeze(-1)

