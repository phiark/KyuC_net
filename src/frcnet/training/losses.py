from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from frcnet.data import BatchInput, validate_batch_input
from frcnet.models import ModelOutput


@dataclass(slots=True)
class LossConfig:
    weight_id: float = 1.0
    weight_unknown: float = 1.0
    weight_ambiguous: float = 1.0
    ambiguous_resolution_target: float = 0.8
    ambiguous_resolution_weight: float = 1.0


@dataclass(slots=True)
class LossBreakdown:
    loss_id: torch.Tensor
    loss_unknown: torch.Tensor
    loss_ambiguous: torch.Tensor
    loss_total: torch.Tensor
    num_trainable_samples: int
    optimizer_step_performed: bool = False


def _cohort_mask(cohort_names: list[str], accepted_names: set[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([name in accepted_names for name in cohort_names], device=device, dtype=torch.bool)


def _safe_log(probabilities: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(probabilities.dtype).eps
    return torch.log(probabilities.clamp_min(epsilon))


def _connected_zero(reference_tensor: torch.Tensor) -> torch.Tensor:
    return reference_tensor * 0.0


def _normalize_loss_config(loss_config: LossConfig | Mapping[str, float] | None) -> LossConfig:
    if loss_config is None:
        return LossConfig()
    if isinstance(loss_config, LossConfig):
        return loss_config
    return LossConfig(**loss_config)


def compute_total_loss(
    model_output: ModelOutput,
    batch_input: BatchInput,
    loss_config: LossConfig | Mapping[str, float] | None = None,
) -> LossBreakdown:
    validate_batch_input(batch_input, num_classes=model_output.num_classes)
    resolved_config = _normalize_loss_config(loss_config)
    reference = model_output.class_mass.sum()
    device = model_output.class_mass.device

    id_mask = _cohort_mask(batch_input.cohort_name, {"easy_id", "hard_id"}, device)
    ambiguous_mask = _cohort_mask(batch_input.cohort_name, {"ambiguous_id"}, device)
    unknown_mask = _cohort_mask(batch_input.cohort_name, {"unknown_supervision"}, device)

    loss_id = _connected_zero(reference)
    if bool(id_mask.any()):
        target_index = batch_input.class_label[id_mask].long()
        selected_mass = model_output.class_mass[id_mask].gather(1, target_index.unsqueeze(1)).squeeze(1)
        loss_id = (-_safe_log(selected_mass)).mean()

    loss_unknown = _connected_zero(reference)
    if bool(unknown_mask.any()):
        selected_unknown_mass = model_output.unknown_mass[unknown_mask]
        loss_unknown = (-_safe_log(selected_unknown_mass)).mean()

    loss_ambiguous = _connected_zero(reference)
    if bool(ambiguous_mask.any()):
        if batch_input.candidate_class_mask is None:
            raise ValueError("candidate_class_mask is required for ambiguous_id samples.")
        candidate_class_mask = batch_input.candidate_class_mask[ambiguous_mask].to(
            device=device, dtype=model_output.content_distribution.dtype
        )
        candidate_count = candidate_class_mask.sum(dim=1, keepdim=True)
        if torch.any(candidate_count.eq(0)):
            raise ValueError("Each ambiguous_id sample must expose at least one candidate class.")
        target_distribution = candidate_class_mask / candidate_count
        content_log_probability = _safe_log(model_output.content_distribution[ambiguous_mask])
        content_loss = -(target_distribution * content_log_probability).sum(dim=1).mean()
        resolution_loss = (
            model_output.resolution_ratio[ambiguous_mask] - resolved_config.ambiguous_resolution_target
        ).pow(2).mean()
        loss_ambiguous = content_loss + (resolved_config.ambiguous_resolution_weight * resolution_loss)

    loss_total = (
        (resolved_config.weight_id * loss_id)
        + (resolved_config.weight_unknown * loss_unknown)
        + (resolved_config.weight_ambiguous * loss_ambiguous)
    )
    num_trainable_samples = int(id_mask.sum().item() + ambiguous_mask.sum().item() + unknown_mask.sum().item())
    return LossBreakdown(
        loss_id=loss_id,
        loss_unknown=loss_unknown,
        loss_ambiguous=loss_ambiguous,
        loss_total=loss_total,
        num_trainable_samples=num_trainable_samples,
    )
