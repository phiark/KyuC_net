from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as tvf

from frcnet.data.contracts import BatchInput
from frcnet.data.manifest import SampleManifestRecord

CIFAR10_CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(slots=True)
class ManifestSample:
    image: torch.Tensor
    class_label: int
    sample_id: str
    split_name: str
    cohort_name: str
    source_dataset_name: str
    source_class_label: int | None
    candidate_class_mask: torch.Tensor | None


def _extract_labels(dataset: object) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(label) for label in getattr(dataset, "targets")]
    if hasattr(dataset, "labels"):
        return [int(label) for label in getattr(dataset, "labels")]
    raise ValueError("Dataset must expose either `targets` or `labels`.")


def load_plan_a_source_datasets(protocol_config: Mapping[str, Any]) -> dict[str, object]:
    datasets_config = protocol_config["datasets"]
    cifar_config = datasets_config["cifar10"]
    svhn_config = datasets_config["svhn"]
    return {
        "cifar10": datasets.CIFAR10(
            root=cifar_config["root"],
            train=bool(cifar_config.get("train", False)),
            download=bool(cifar_config.get("download", False)),
        ),
        "svhn": datasets.SVHN(
            root=svhn_config["root"],
            split=svhn_config.get("split", "test"),
            download=bool(svhn_config.get("download", False)),
        ),
    }


def _labels_to_class_index(labels: Iterable[int]) -> dict[int, list[int]]:
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for sample_index, label in enumerate(labels):
        class_to_indices[int(label)].append(sample_index)
    return class_to_indices


def _pop_indices(index_pool: list[int], count: int) -> list[int]:
    if len(index_pool) < count:
        raise ValueError("Insufficient indices to satisfy manifest allocation.")
    selected = index_pool[:count]
    del index_pool[:count]
    return selected


def _stable_recipe(index: int) -> tuple[str, dict[str, Any]]:
    if index % 2 == 0:
        return "gaussian_blur", {"kernel_size": 5, "sigma": 1.0}
    return "low_res", {"downsample_size": 16}


def build_plan_a_manifest(
    protocol_config: Mapping[str, Any],
    source_datasets: Mapping[str, object],
) -> list[SampleManifestRecord]:
    protocol_id = protocol_config["protocol_id"]
    seed = int(protocol_config.get("seed", 7))
    split_name = protocol_config.get("split_name", "analysis")
    rng = random.Random(seed)

    cifar_labels = _extract_labels(source_datasets["cifar10"])
    svhn_labels = _extract_labels(source_datasets["svhn"])

    cifar_indices = _labels_to_class_index(cifar_labels)
    for label_indices in cifar_indices.values():
        rng.shuffle(label_indices)
    svhn_indices = list(range(len(svhn_labels)))
    rng.shuffle(svhn_indices)

    manifest_records: list[SampleManifestRecord] = []
    analysis_config = protocol_config["analysis"]
    easy_id_per_class = int(analysis_config["easy_id_per_class"])
    hard_id_per_class = int(analysis_config["hard_id_per_class"])
    ambiguous_per_pair = int(analysis_config["ambiguous_per_pair"])
    ood_count = int(analysis_config["ood_count"])
    unknown_count = int(analysis_config["unknown_supervision_count"])

    for class_label in range(int(protocol_config["num_classes"])):
        easy_indices = _pop_indices(cifar_indices[class_label], easy_id_per_class)
        for index in easy_indices:
            manifest_records.append(
                SampleManifestRecord(
                    protocol_id=protocol_id,
                    sample_id=f"{split_name}_easy_id_cifar10_{index:05d}",
                    split_name=split_name,
                    cohort_name="easy_id",
                    source_dataset_name="cifar10",
                    source_sample_indices=(index,),
                    source_class_label=class_label,
                    class_label=class_label,
                    augmentation_recipe="identity",
                    augmentation_parameters={},
                    source_class_labels=(class_label,),
                )
            )

        hard_indices = _pop_indices(cifar_indices[class_label], hard_id_per_class)
        for hard_offset, index in enumerate(hard_indices):
            recipe, parameters = _stable_recipe(hard_offset)
            manifest_records.append(
                SampleManifestRecord(
                    protocol_id=protocol_id,
                    sample_id=f"{split_name}_hard_id_cifar10_{index:05d}",
                    split_name=split_name,
                    cohort_name="hard_id",
                    source_dataset_name="cifar10",
                    source_sample_indices=(index,),
                    source_class_label=class_label,
                    class_label=class_label,
                    augmentation_recipe=recipe,
                    augmentation_parameters=parameters,
                    source_class_labels=(class_label,),
                )
            )

    alpha_min = float(protocol_config["ambiguous"]["alpha_min"])
    alpha_max = float(protocol_config["ambiguous"]["alpha_max"])
    class_pairs = [tuple(int(class_index) for class_index in pair) for pair in protocol_config["ambiguous"]["class_pairs"]]
    for pair_index, class_pair in enumerate(class_pairs):
        left_class, right_class = class_pair
        left_indices = _pop_indices(cifar_indices[left_class], ambiguous_per_pair)
        right_indices = _pop_indices(cifar_indices[right_class], ambiguous_per_pair)
        for ambiguous_index, (left_index, right_index) in enumerate(zip(left_indices, right_indices, strict=True)):
            alpha = alpha_min if ambiguous_per_pair == 1 else alpha_min + (
                (alpha_max - alpha_min) * ambiguous_index / (ambiguous_per_pair - 1)
            )
            manifest_records.append(
                SampleManifestRecord(
                    protocol_id=protocol_id,
                    sample_id=f"{split_name}_ambiguous_{pair_index}_{ambiguous_index:03d}",
                    split_name=split_name,
                    cohort_name="ambiguous_id",
                    source_dataset_name="cifar10",
                    source_sample_indices=(left_index, right_index),
                    source_class_label=None,
                    class_label=-1,
                    candidate_class_indices=class_pair,
                    augmentation_recipe="mixup",
                    augmentation_parameters={"alpha": alpha},
                    source_class_labels=class_pair,
                )
            )

    ood_indices = _pop_indices(svhn_indices, ood_count)
    for index in ood_indices:
        manifest_records.append(
            SampleManifestRecord(
                protocol_id=protocol_id,
                sample_id=f"{split_name}_ood_svhn_{index:05d}",
                split_name=split_name,
                cohort_name="ood",
                source_dataset_name="svhn",
                source_sample_indices=(index,),
                source_class_label=int(svhn_labels[index]),
                class_label=-1,
                augmentation_recipe="identity",
                augmentation_parameters={},
                source_class_labels=(int(svhn_labels[index]),),
            )
        )

    unknown_indices = _pop_indices(svhn_indices, unknown_count)
    for index in unknown_indices:
        manifest_records.append(
            SampleManifestRecord(
                protocol_id=protocol_id,
                sample_id=f"{split_name}_unknown_svhn_{index:05d}",
                split_name=split_name,
                cohort_name="unknown_supervision",
                source_dataset_name="svhn",
                source_sample_indices=(index,),
                source_class_label=int(svhn_labels[index]),
                class_label=-1,
                augmentation_recipe="identity",
                augmentation_parameters={},
                source_class_labels=(int(svhn_labels[index]),),
            )
        )

    return manifest_records


def _to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if tensor.ndim == 3 and tensor.dtype.is_floating_point:
            return tensor
        if tensor.ndim == 3:
            return tensor.float() / 255.0
        raise ValueError("Expected image tensors to use CHW layout.")
    return tvf.to_tensor(image)


def _load_record_image(record: SampleManifestRecord, source_datasets: Mapping[str, object]) -> torch.Tensor:
    dataset = source_datasets[record.source_dataset_name]
    if record.augmentation_recipe == "mixup":
        left_image, _ = dataset[record.source_sample_indices[0]]
        right_image, _ = dataset[record.source_sample_indices[1]]
        alpha = float(record.augmentation_parameters["alpha"])
        return (alpha * _to_tensor(left_image)) + ((1.0 - alpha) * _to_tensor(right_image))

    image, _ = dataset[record.source_sample_indices[0]]
    tensor = _to_tensor(image)
    if record.augmentation_recipe == "gaussian_blur":
        kernel_size = int(record.augmentation_parameters.get("kernel_size", 5))
        sigma = float(record.augmentation_parameters.get("sigma", 1.0))
        return tvf.gaussian_blur(tensor, [kernel_size, kernel_size], [sigma, sigma])
    if record.augmentation_recipe == "low_res":
        downsample_size = int(record.augmentation_parameters.get("downsample_size", 16))
        original_height, original_width = int(tensor.shape[1]), int(tensor.shape[2])
        low_res = tvf.resize(tensor, [downsample_size, downsample_size], antialias=True)
        return tvf.resize(low_res, [original_height, original_width], antialias=True)
    return tensor


class ManifestBackedVisionDataset(Dataset[ManifestSample]):
    def __init__(
        self,
        manifest_records: list[SampleManifestRecord],
        source_datasets: Mapping[str, object],
        num_classes: int,
    ) -> None:
        self.manifest_records = manifest_records
        self.source_datasets = source_datasets
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.manifest_records)

    def __getitem__(self, index: int) -> ManifestSample:
        record = self.manifest_records[index]
        candidate_class_mask = None
        if record.candidate_class_indices:
            candidate_class_mask = torch.zeros(self.num_classes, dtype=torch.bool)
            candidate_class_mask[list(record.candidate_class_indices)] = True

        return ManifestSample(
            image=_load_record_image(record, self.source_datasets),
            class_label=record.class_label,
            sample_id=record.sample_id,
            split_name=record.split_name,
            cohort_name=record.cohort_name,
            source_dataset_name=record.source_dataset_name,
            source_class_label=record.source_class_label,
            candidate_class_mask=candidate_class_mask,
        )


def collate_manifest_samples(samples: list[ManifestSample]) -> BatchInput:
    candidate_class_mask = None
    if any(sample.candidate_class_mask is not None for sample in samples):
        template_mask = next(
            sample.candidate_class_mask for sample in samples if sample.candidate_class_mask is not None
        )
        mask_tensors = []
        for sample in samples:
            if sample.candidate_class_mask is None:
                mask_tensors.append(torch.zeros_like(template_mask))
            else:
                mask_tensors.append(sample.candidate_class_mask)
        candidate_class_mask = torch.stack(mask_tensors, dim=0)

    return BatchInput(
        image=torch.stack([sample.image for sample in samples], dim=0),
        class_label=torch.tensor([sample.class_label for sample in samples], dtype=torch.long),
        sample_id=[sample.sample_id for sample in samples],
        split_name=[sample.split_name for sample in samples],
        cohort_name=[sample.cohort_name for sample in samples],
        source_dataset_name=[sample.source_dataset_name for sample in samples],
        source_class_label=[sample.source_class_label for sample in samples],
        candidate_class_mask=candidate_class_mask,
    )


def summarize_manifest(records: Iterable[SampleManifestRecord]) -> dict[str, Any]:
    cohort_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for record in records:
        cohort_counts[record.cohort_name] += 1
        split_counts[record.split_name] += 1
    return {
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }


def write_manifest_summary(records: Iterable[SampleManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_manifest(records), indent=2, sort_keys=True), encoding="utf-8")
    return output
