from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

ALLOWED_COHORT_NAMES: Final[frozenset[str]] = frozenset(
    {"easy_id", "ambiguous_id", "hard_id", "ood", "unknown_supervision"}
)


@dataclass(slots=True)
class BatchInput:
    image: torch.Tensor
    class_label: torch.Tensor
    sample_id: list[str]
    split_name: list[str]
    cohort_name: list[str]
    source_dataset_name: list[str]
    source_class_label: list[int | None] | None = None
    candidate_class_mask: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.image.shape[0])


def validate_batch_input(batch_input: BatchInput, num_classes: int | None = None) -> None:
    batch_size = batch_input.batch_size

    if batch_input.class_label.ndim != 1 or int(batch_input.class_label.shape[0]) != batch_size:
        raise ValueError("class_label must be a 1D tensor aligned with the batch dimension.")

    for field_name, field_value in (
        ("sample_id", batch_input.sample_id),
        ("split_name", batch_input.split_name),
        ("cohort_name", batch_input.cohort_name),
        ("source_dataset_name", batch_input.source_dataset_name),
    ):
        if len(field_value) != batch_size:
            raise ValueError(f"{field_name} must contain one entry per batch item.")

    unknown_cohorts = sorted(set(batch_input.cohort_name) - ALLOWED_COHORT_NAMES)
    if unknown_cohorts:
        raise ValueError(f"Unsupported cohort names: {unknown_cohorts}")

    if torch.any(batch_input.class_label < -1):
        raise ValueError("class_label entries must be >= -1.")

    if batch_input.source_class_label is not None and len(batch_input.source_class_label) != batch_size:
        raise ValueError("source_class_label must contain one entry per batch item when provided.")

    ambiguous_indices = [index for index, cohort in enumerate(batch_input.cohort_name) if cohort == "ambiguous_id"]
    id_indices = [index for index, cohort in enumerate(batch_input.cohort_name) if cohort in {"easy_id", "hard_id"}]
    ood_indices = [index for index, cohort in enumerate(batch_input.cohort_name) if cohort == "ood"]
    unknown_indices = [
        index for index, cohort in enumerate(batch_input.cohort_name) if cohort == "unknown_supervision"
    ]

    if id_indices:
        id_labels = batch_input.class_label[id_indices]
        if torch.any(id_labels.eq(-1)):
            raise ValueError("easy_id and hard_id samples must use in-domain class labels.")
        if num_classes is not None:
            if torch.any(id_labels >= num_classes):
                raise ValueError("easy_id and hard_id labels must be within [0, num_classes).")

    if ambiguous_indices:
        ambiguous_labels = batch_input.class_label[ambiguous_indices]
        if not torch.all(ambiguous_labels.eq(-1)):
            raise ValueError("ambiguous_id samples must use class_label = -1 and rely on candidate_class_mask.")

    if unknown_indices:
        unknown_labels = batch_input.class_label[unknown_indices]
        if not torch.all(unknown_labels.eq(-1)):
            raise ValueError("unknown_supervision samples must use class_label = -1.")

    if ood_indices:
        ood_labels = batch_input.class_label[ood_indices]
        if not torch.all(ood_labels.eq(-1)):
            raise ValueError("ood samples must use class_label = -1.")

    if ambiguous_indices and batch_input.candidate_class_mask is None:
        raise ValueError("ambiguous_id samples require candidate_class_mask.")

    if batch_input.candidate_class_mask is not None:
        candidate_mask = batch_input.candidate_class_mask
        if candidate_mask.ndim != 2 or int(candidate_mask.shape[0]) != batch_size:
            raise ValueError("candidate_class_mask must be a 2D tensor aligned with the batch dimension.")
        if candidate_mask.dtype is not torch.bool:
            raise ValueError("candidate_class_mask must use torch.bool dtype.")
        if num_classes is not None and int(candidate_mask.shape[1]) != num_classes:
            raise ValueError("candidate_class_mask width must match num_classes.")
        if ambiguous_indices:
            ambiguous_mask = candidate_mask[ambiguous_indices]
            if torch.any(ambiguous_mask.sum(dim=1) == 0):
                raise ValueError("Each ambiguous_id sample must expose at least one candidate class.")
