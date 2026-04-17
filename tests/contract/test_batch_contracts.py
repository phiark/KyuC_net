from __future__ import annotations

import pytest
import torch

from frcnet.data import BatchInput, validate_batch_input
from frcnet.utils import move_batch_to_device, resolve_runtime
from tests.conftest import build_synthetic_batch


def test_validate_batch_input_accepts_standard_cohorts():
    batch_input = build_synthetic_batch()

    validate_batch_input(batch_input, num_classes=10)


def test_validate_batch_input_rejects_unknown_cohort():
    batch_input = build_synthetic_batch()
    batch_input.cohort_name[0] = "mystery"

    with pytest.raises(ValueError):
        validate_batch_input(batch_input, num_classes=10)


def test_move_batch_to_device_preserves_metadata():
    batch_input = build_synthetic_batch()
    runtime_spec = resolve_runtime(requested_backend="cpu")

    moved_batch = move_batch_to_device(batch_input, runtime_spec)

    assert moved_batch.image.device.type == "cpu"
    assert moved_batch.sample_id == batch_input.sample_id
    assert moved_batch.cohort_name == batch_input.cohort_name
    assert moved_batch.source_dataset_name == batch_input.source_dataset_name
    assert moved_batch.candidate_class_mask is not None
    assert moved_batch.candidate_class_mask.dtype is torch.bool


def test_validate_batch_input_requires_candidate_mask_for_ambiguous():
    batch_input = build_synthetic_batch()
    missing_candidate_batch = BatchInput(
        image=batch_input.image,
        class_label=batch_input.class_label,
        sample_id=batch_input.sample_id,
        split_name=batch_input.split_name,
        cohort_name=batch_input.cohort_name,
        source_dataset_name=batch_input.source_dataset_name,
        candidate_class_mask=None,
    )

    with pytest.raises(ValueError):
        validate_batch_input(missing_candidate_batch, num_classes=10)

