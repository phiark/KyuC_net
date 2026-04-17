from __future__ import annotations

import torch

from frcnet.data import BatchInput


def build_synthetic_batch(num_classes: int = 10) -> BatchInput:
    batch_size = 5
    candidate_class_mask = torch.zeros((batch_size, num_classes), dtype=torch.bool)
    candidate_class_mask[2, 2] = True
    candidate_class_mask[2, 3] = True

    return BatchInput(
        image=torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([1, 4, 2, -1, 6], dtype=torch.long),
        sample_id=[f"sample-{index}" for index in range(batch_size)],
        split_name=["train"] * batch_size,
        cohort_name=["easy_id", "hard_id", "ambiguous_id", "unknown_supervision", "ood"],
        source_dataset_name=["synthetic"] * batch_size,
        candidate_class_mask=candidate_class_mask,
    )

