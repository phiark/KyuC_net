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
        class_label=torch.tensor([1, 4, -1, -1, -1], dtype=torch.long),
        sample_id=[f"sample-{index}" for index in range(batch_size)],
        split_name=["train"] * batch_size,
        cohort_name=["easy_id", "hard_id", "ambiguous_id", "unknown_supervision", "ood"],
        source_dataset_name=["synthetic"] * batch_size,
        source_class_label=[1, 4, None, 7, 6],
        candidate_class_mask=candidate_class_mask,
    )


class FakeTargetsDataset:
    def __init__(self, labels: list[int]) -> None:
        self.targets = labels
        self.images = [torch.rand(3, 32, 32, dtype=torch.float32) for _ in labels]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


class FakeLabelsDataset:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.images = [torch.rand(3, 32, 32, dtype=torch.float32) for _ in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def build_protocol_config() -> dict:
    return {
        "protocol_id": "plan_a_v1",
        "seed": 7,
        "split_name": "analysis",
        "num_classes": 10,
        "analysis": {
            "easy_id_per_class": 1,
            "hard_id_per_class": 1,
            "ambiguous_per_pair": 1,
            "ood_count": 2,
            "unknown_supervision_count": 2,
            "dataloader": {
                "batch_size": 4,
                "drop_last": True,
                "num_workers": 0,
            },
        },
        "ambiguous": {
            "alpha_min": 0.35,
            "alpha_max": 0.65,
            "class_pairs": [[3, 5], [4, 7], [1, 9]],
        },
        "datasets": {
            "cifar10": {"root": "data/cifar10", "train": False, "download": False},
            "svhn": {"root": "data/svhn", "split": "test", "download": False},
        },
    }
