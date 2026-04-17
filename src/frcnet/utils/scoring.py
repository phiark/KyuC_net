from __future__ import annotations

import torch


def content_entropy(content_distribution: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(content_distribution.dtype).eps
    safe_distribution = content_distribution.clamp_min(epsilon)
    return -(safe_distribution * torch.log(safe_distribution)).sum(dim=-1)


def top1_class_mass(class_mass: torch.Tensor) -> torch.Tensor:
    return class_mass.max(dim=-1).values


def completion_score(class_mass: torch.Tensor, unknown_mass: torch.Tensor, beta: float) -> torch.Tensor:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be within [0, 1].")
    return top1_class_mass(class_mass) + (beta * unknown_mass)

