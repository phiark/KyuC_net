from __future__ import annotations

import torch


def top1_symmetric_beta(num_classes: int) -> float:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    return 1.0 / float(num_classes)


def candidate_symmetric_beta(candidate_size: int, num_classes: int) -> float:
    if candidate_size < 0:
        raise ValueError("candidate_size must be >= 0.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if candidate_size > num_classes:
        raise ValueError("candidate_size cannot exceed num_classes.")
    return float(candidate_size) / float(num_classes)


def binary_pignistic_beta() -> float:
    return 0.5


def completion_from_view(
    truth_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be within [0, 1].")
    return truth_mass + (beta * unknown_mass)
