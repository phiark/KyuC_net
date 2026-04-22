from __future__ import annotations

import torch


def _safe_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(probabilities.dtype).eps
    return probabilities.clamp_min(epsilon)


def content_entropy(content_distribution: torch.Tensor) -> torch.Tensor:
    safe_distribution = _safe_distribution(content_distribution)
    return -(safe_distribution * torch.log(safe_distribution)).sum(dim=-1)


def top1_class_mass(class_mass: torch.Tensor) -> torch.Tensor:
    return class_mass.max(dim=-1).values


def resolution_weighted_content_entropy(
    resolution_ratio: torch.Tensor,
    content_entropy_value: torch.Tensor,
) -> torch.Tensor:
    return resolution_ratio * content_entropy_value


def resolution_entropy(resolution_ratio: torch.Tensor) -> torch.Tensor:
    probability_stack = torch.stack((resolution_ratio, 1.0 - resolution_ratio), dim=-1)
    safe_distribution = _safe_distribution(probability_stack)
    return -(safe_distribution * torch.log(safe_distribution)).sum(dim=-1)


def ternary_entropy_from_masses(
    truth_mass: torch.Tensor,
    false_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
) -> torch.Tensor:
    probability_stack = torch.stack((truth_mass, false_mass, unknown_mass), dim=-1)
    safe_distribution = _safe_distribution(probability_stack)
    return -(safe_distribution * torch.log(safe_distribution)).sum(dim=-1)


def completion_score(class_mass: torch.Tensor, unknown_mass: torch.Tensor, beta: float) -> torch.Tensor:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be within [0, 1].")
    return top1_class_mass(class_mass) + (beta * unknown_mass)
