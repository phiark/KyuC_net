from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from frcnet.data import BatchInput, validate_batch_input
from frcnet.models import ModelOutput
from frcnet.utils import content_entropy


@dataclass(slots=True)
class LossConfig:
    weight_id: float = 1.0
    weight_unknown: float = 1.0
    weight_ambiguous: float = 1.0
    unknown_content_entropy_weight: float = 0.0
    hard_id_label_smoothing: float = 0.0
    hard_id_resolution_floor: float = 0.0
    hard_id_resolution_weight: float = 0.0
    hard_id_entropy_ceiling: float = 0.0
    hard_id_entropy_weight: float = 0.0
    ambiguous_entropy_floor_margin: float = 0.0
    ambiguous_entropy_floor_weight: float = 0.0
    ambiguous_resolution_target: float = 0.8
    ambiguous_resolution_weight: float = 1.0


@dataclass(slots=True)
class LossBreakdown:
    loss_id: torch.Tensor
    loss_unknown: torch.Tensor
    loss_unknown_content: torch.Tensor
    loss_ambiguous: torch.Tensor
    loss_hard_resolution_floor: torch.Tensor
    loss_hard_entropy_ceiling: torch.Tensor
    loss_ambiguous_entropy_floor: torch.Tensor
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


def _smoothed_target_distribution(
    target_index: torch.Tensor,
    *,
    num_classes: int,
    smoothing: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("hard_id_label_smoothing must be within [0, 1).")
    if num_classes <= 1 or smoothing == 0.0:
        return torch.nn.functional.one_hot(target_index, num_classes=num_classes).to(dtype=dtype)

    off_value = smoothing / (num_classes - 1)
    target_distribution = torch.full(
        (target_index.shape[0], num_classes),
        fill_value=off_value,
        device=target_index.device,
        dtype=dtype,
    )
    target_distribution.scatter_(1, target_index.unsqueeze(1), 1.0 - smoothing)
    return target_distribution


def compute_total_loss(
    model_output: ModelOutput,
    batch_input: BatchInput,
    loss_config: LossConfig | Mapping[str, float] | None = None,
) -> LossBreakdown:
    validate_batch_input(batch_input, num_classes=model_output.num_classes)
    resolved_config = _normalize_loss_config(loss_config)
    if resolved_config.unknown_content_entropy_weight < 0.0:
        raise ValueError("unknown_content_entropy_weight must be >= 0.")
    if resolved_config.hard_id_resolution_weight < 0.0:
        raise ValueError("hard_id_resolution_weight must be >= 0.")
    if resolved_config.hard_id_entropy_weight < 0.0:
        raise ValueError("hard_id_entropy_weight must be >= 0.")
    if resolved_config.ambiguous_entropy_floor_weight < 0.0:
        raise ValueError("ambiguous_entropy_floor_weight must be >= 0.")
    reference = model_output.class_mass.sum()
    device = model_output.class_mass.device

    easy_id_mask = _cohort_mask(batch_input.cohort_name, {"easy_id"}, device)
    hard_id_mask = _cohort_mask(batch_input.cohort_name, {"hard_id"}, device)
    ambiguous_mask = _cohort_mask(batch_input.cohort_name, {"ambiguous_id"}, device)
    unknown_mask = _cohort_mask(batch_input.cohort_name, {"unknown_supervision"}, device)

    loss_id = _connected_zero(reference)
    id_loss_terms: list[torch.Tensor] = []
    if bool(easy_id_mask.any()):
        target_index = batch_input.class_label[easy_id_mask].long()
        selected_mass = model_output.class_mass[easy_id_mask].gather(1, target_index.unsqueeze(1)).squeeze(1)
        id_loss_terms.append(-_safe_log(selected_mass))
    if bool(hard_id_mask.any()):
        target_index = batch_input.class_label[hard_id_mask].long()
        smoothed_distribution = _smoothed_target_distribution(
            target_index,
            num_classes=model_output.num_classes,
            smoothing=float(resolved_config.hard_id_label_smoothing),
            dtype=model_output.class_mass.dtype,
        )
        hard_class_log_probability = _safe_log(model_output.class_mass[hard_id_mask])
        id_loss_terms.append(-(smoothed_distribution * hard_class_log_probability).sum(dim=1))
    if id_loss_terms:
        loss_id = torch.cat(id_loss_terms).mean()

    loss_unknown = _connected_zero(reference)
    if bool(unknown_mask.any()):
        selected_unknown_mass = model_output.unknown_mass[unknown_mask]
        loss_unknown = (-_safe_log(selected_unknown_mass)).mean()

    loss_unknown_content = _connected_zero(reference)
    if bool(unknown_mask.any()) and float(resolved_config.unknown_content_entropy_weight) > 0.0:
        entropy = content_entropy(model_output.content_distribution[unknown_mask])
        max_entropy = torch.log(
            torch.tensor(float(model_output.num_classes), device=device, dtype=model_output.content_distribution.dtype)
        )
        loss_unknown_content = (max_entropy - entropy).mean()

    loss_hard_resolution_floor = _connected_zero(reference)
    if bool(hard_id_mask.any()) and float(resolved_config.hard_id_resolution_weight) > 0.0:
        resolution_floor = torch.tensor(
            float(resolved_config.hard_id_resolution_floor),
            device=device,
            dtype=model_output.resolution_ratio.dtype,
        )
        loss_hard_resolution_floor = torch.relu(
            resolution_floor - model_output.resolution_ratio[hard_id_mask]
        ).pow(2).mean()

    loss_hard_entropy_ceiling = _connected_zero(reference)
    if bool(hard_id_mask.any()) and float(resolved_config.hard_id_entropy_weight) > 0.0:
        entropy_ceiling = torch.tensor(
            float(resolved_config.hard_id_entropy_ceiling),
            device=device,
            dtype=model_output.content_distribution.dtype,
        )
        hard_entropy = content_entropy(model_output.content_distribution[hard_id_mask])
        loss_hard_entropy_ceiling = torch.relu(hard_entropy - entropy_ceiling).pow(2).mean()

    loss_ambiguous = _connected_zero(reference)
    loss_ambiguous_entropy_floor = _connected_zero(reference)
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
        ambiguous_entropy = content_entropy(model_output.content_distribution[ambiguous_mask])
        content_loss = -(target_distribution * content_log_probability).sum(dim=1).mean()
        resolution_loss = (
            model_output.resolution_ratio[ambiguous_mask] - resolved_config.ambiguous_resolution_target
        ).pow(2).mean()
        loss_ambiguous = content_loss + (resolved_config.ambiguous_resolution_weight * resolution_loss)
        if float(resolved_config.ambiguous_entropy_floor_weight) > 0.0:
            entropy_floor = torch.log(candidate_count.squeeze(1)).to(dtype=ambiguous_entropy.dtype) - float(
                resolved_config.ambiguous_entropy_floor_margin
            )
            loss_ambiguous_entropy_floor = torch.relu(entropy_floor - ambiguous_entropy).pow(2).mean()

    loss_total = (
        (resolved_config.weight_id * loss_id)
        + (resolved_config.weight_unknown * loss_unknown)
        + (resolved_config.weight_ambiguous * loss_ambiguous)
        + (resolved_config.unknown_content_entropy_weight * loss_unknown_content)
        + (resolved_config.hard_id_resolution_weight * loss_hard_resolution_floor)
        + (resolved_config.hard_id_entropy_weight * loss_hard_entropy_ceiling)
        + (resolved_config.ambiguous_entropy_floor_weight * loss_ambiguous_entropy_floor)
    )
    num_trainable_samples = int(
        easy_id_mask.sum().item() + hard_id_mask.sum().item() + ambiguous_mask.sum().item() + unknown_mask.sum().item()
    )
    return LossBreakdown(
        loss_id=loss_id,
        loss_unknown=loss_unknown,
        loss_unknown_content=loss_unknown_content,
        loss_ambiguous=loss_ambiguous,
        loss_hard_resolution_floor=loss_hard_resolution_floor,
        loss_hard_entropy_ceiling=loss_hard_entropy_ceiling,
        loss_ambiguous_entropy_floor=loss_ambiguous_entropy_floor,
        loss_total=loss_total,
        num_trainable_samples=num_trainable_samples,
    )
