from __future__ import annotations

import torch

from frcnet.utils import content_entropy


def softmax_entropy_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return content_entropy(probabilities)
