from __future__ import annotations

import torch

from frcnet.utils import content_entropy, resolution_entropy


def state_content_entropy(content_distribution: torch.Tensor) -> torch.Tensor:
    return content_entropy(content_distribution)


def state_weighted_content_entropy(
    resolution_ratio: torch.Tensor,
    state_content_entropy_value: torch.Tensor,
) -> torch.Tensor:
    return resolution_ratio * state_content_entropy_value


def state_entropy(
    resolution_ratio: torch.Tensor,
    state_content_entropy_value: torch.Tensor,
) -> torch.Tensor:
    return resolution_entropy(resolution_ratio) + state_weighted_content_entropy(
        resolution_ratio,
        state_content_entropy_value,
    )
