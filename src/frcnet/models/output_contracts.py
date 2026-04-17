from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelOutput:
    backbone_feature: torch.Tensor
    resolution_logit: torch.Tensor
    resolution_ratio: torch.Tensor
    content_logits: torch.Tensor
    content_distribution: torch.Tensor
    class_mass: torch.Tensor
    unknown_mass: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.class_mass.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.class_mass.shape[1])

