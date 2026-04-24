from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PropositionViewTensor:
    view_name: str
    label_aware: bool
    truth_mass: torch.Tensor
    false_mass: torch.Tensor
    unknown_mass: torch.Tensor
    tau: torch.Tensor


def proposition_view_from_mask(
    *,
    view_name: str,
    label_aware: bool,
    class_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
    truth_mask: torch.Tensor,
) -> PropositionViewTensor:
    if class_mass.shape != truth_mask.shape:
        raise ValueError("class_mass and truth_mask must have the same shape.")
    truth_mask_value = truth_mask.to(dtype=class_mass.dtype, device=class_mass.device)
    truth_mass = (class_mass * truth_mask_value).sum(dim=1)
    resolved_mass = class_mass.sum(dim=1)
    false_mass = (resolved_mass - truth_mass).clamp_min(0.0)
    tau = torch.zeros_like(truth_mass)
    resolved_mask = resolved_mass > torch.finfo(resolved_mass.dtype).eps
    tau[resolved_mask] = truth_mass[resolved_mask] / resolved_mass[resolved_mask]
    return PropositionViewTensor(
        view_name=view_name,
        label_aware=label_aware,
        truth_mass=truth_mass,
        false_mass=false_mass,
        unknown_mass=unknown_mass,
        tau=tau,
    )


def top1_view(class_mass: torch.Tensor, unknown_mass: torch.Tensor) -> PropositionViewTensor:
    top1_index = torch.argmax(class_mass, dim=1)
    truth_mask = torch.zeros_like(class_mass, dtype=torch.bool)
    truth_mask.scatter_(1, top1_index.unsqueeze(1), True)
    return proposition_view_from_mask(
        view_name="top1_view",
        label_aware=False,
        class_mass=class_mass,
        unknown_mass=unknown_mass,
        truth_mask=truth_mask,
    )
