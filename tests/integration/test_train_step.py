from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from frcnet.data import BatchInput
from frcnet.models import FRCNetModel
from frcnet.training import compute_total_loss, run_train_step
from frcnet.utils import resolve_runtime
from tests.conftest import build_synthetic_batch

HAS_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
HAS_ROCM = bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)
HAS_CUDA = bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None)


def test_compute_total_loss_routes_all_training_cohorts():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    loss_breakdown = compute_total_loss(model_output, batch_input)

    assert loss_breakdown.loss_total.requires_grad
    assert torch.isfinite(loss_breakdown.loss_id)
    assert torch.isfinite(loss_breakdown.loss_unknown)
    assert torch.isfinite(loss_breakdown.loss_unknown_content)
    assert torch.isfinite(loss_breakdown.loss_ambiguous)
    assert torch.isfinite(loss_breakdown.loss_hard_resolution_floor)
    assert torch.isfinite(loss_breakdown.loss_hard_entropy_ceiling)
    assert torch.isfinite(loss_breakdown.loss_ambiguous_entropy_floor)
    assert torch.isfinite(loss_breakdown.loss_total)


def test_compute_total_loss_unknown_content_regularizer_penalizes_peaked_unknown_content():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    peaked_distribution = model_output.content_distribution.clone()
    peaked_distribution[3] = torch.tensor(
        [0.97] + [0.0033333334] * 9,
        dtype=peaked_distribution.dtype,
    )
    uniform_distribution = model_output.content_distribution.clone()
    uniform_distribution[3] = torch.full(
        (10,),
        0.1,
        dtype=uniform_distribution.dtype,
    )
    peaked_output = replace(model_output, content_distribution=peaked_distribution)
    uniform_output = replace(model_output, content_distribution=uniform_distribution)

    peaked_loss = compute_total_loss(
        peaked_output,
        batch_input,
        {"unknown_content_entropy_weight": 1.0},
    )
    uniform_loss = compute_total_loss(
        uniform_output,
        batch_input,
        {"unknown_content_entropy_weight": 1.0},
    )

    assert peaked_loss.loss_unknown_content > uniform_loss.loss_unknown_content


def test_run_train_step_cpu_smoke():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)

    assert torch.isfinite(loss_breakdown.loss_total)
    assert loss_breakdown.optimizer_step_performed is True


def test_compute_total_loss_hard_resolution_floor_penalizes_low_resolution_hard_id():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    low_resolution = model_output.resolution_ratio.clone()
    high_resolution = model_output.resolution_ratio.clone()
    hard_index = batch_input.cohort_name.index("hard_id")
    low_resolution[hard_index] = 0.35
    high_resolution[hard_index] = 0.92

    low_output = replace(model_output, resolution_ratio=low_resolution, unknown_mass=1.0 - low_resolution)
    high_output = replace(model_output, resolution_ratio=high_resolution, unknown_mass=1.0 - high_resolution)

    low_loss = compute_total_loss(
        low_output,
        batch_input,
        {"hard_id_resolution_floor": 0.8, "hard_id_resolution_weight": 1.0},
    )
    high_loss = compute_total_loss(
        high_output,
        batch_input,
        {"hard_id_resolution_floor": 0.8, "hard_id_resolution_weight": 1.0},
    )

    assert low_loss.loss_hard_resolution_floor > high_loss.loss_hard_resolution_floor


def test_compute_total_loss_hard_entropy_ceiling_penalizes_high_entropy_hard_id():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    hard_index = batch_input.cohort_name.index("hard_id")

    high_entropy_distribution = model_output.content_distribution.clone()
    high_entropy_distribution[hard_index] = torch.full((10,), 0.1, dtype=high_entropy_distribution.dtype)
    low_entropy_distribution = model_output.content_distribution.clone()
    low_entropy_distribution[hard_index] = torch.tensor(
        [0.97] + [0.0033333334] * 9,
        dtype=low_entropy_distribution.dtype,
    )

    high_entropy_output = replace(model_output, content_distribution=high_entropy_distribution)
    low_entropy_output = replace(model_output, content_distribution=low_entropy_distribution)

    high_entropy_loss = compute_total_loss(
        high_entropy_output,
        batch_input,
        {"hard_id_entropy_ceiling": 1.2, "hard_id_entropy_weight": 1.0},
    )
    low_entropy_loss = compute_total_loss(
        low_entropy_output,
        batch_input,
        {"hard_id_entropy_ceiling": 1.2, "hard_id_entropy_weight": 1.0},
    )

    assert high_entropy_loss.loss_hard_entropy_ceiling > low_entropy_loss.loss_hard_entropy_ceiling


def test_compute_total_loss_ambiguous_entropy_floor_penalizes_overconfident_ambiguous_content():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    ambiguous_index = batch_input.cohort_name.index("ambiguous_id")

    low_entropy_distribution = model_output.content_distribution.clone()
    low_entropy_distribution[ambiguous_index] = torch.tensor(
        [0.97] + [0.0033333334] * 9,
        dtype=low_entropy_distribution.dtype,
    )
    higher_entropy_distribution = model_output.content_distribution.clone()
    higher_entropy_distribution[ambiguous_index] = torch.tensor(
        [0.5, 0.5] + [0.0] * 8,
        dtype=higher_entropy_distribution.dtype,
    )

    low_entropy_output = replace(model_output, content_distribution=low_entropy_distribution)
    higher_entropy_output = replace(model_output, content_distribution=higher_entropy_distribution)

    low_entropy_loss = compute_total_loss(
        low_entropy_output,
        batch_input,
        {"ambiguous_entropy_floor_margin": 0.1, "ambiguous_entropy_floor_weight": 1.0},
    )
    higher_entropy_loss = compute_total_loss(
        higher_entropy_output,
        batch_input,
        {"ambiguous_entropy_floor_margin": 0.1, "ambiguous_entropy_floor_weight": 1.0},
    )

    assert low_entropy_loss.loss_ambiguous_entropy_floor > higher_entropy_loss.loss_ambiguous_entropy_floor


def test_run_train_step_skips_ood_only_batch():
    batch_input = BatchInput(
        image=torch.randn(2, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([-1, -1], dtype=torch.long),
        sample_id=["ood-a", "ood-b"],
        split_name=["train", "train"],
        cohort_name=["ood", "ood"],
        source_dataset_name=["svhn", "svhn"],
        source_class_label=[0, 1],
        candidate_class_mask=None,
    )
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)

    assert loss_breakdown.num_trainable_samples == 0
    assert loss_breakdown.optimizer_step_performed is False
    assert loss_breakdown.loss_total.requires_grad


def test_run_train_step_rejects_singleton_batch_for_batchnorm_backbone():
    batch_input = BatchInput(
        image=torch.randn(1, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([1], dtype=torch.long),
        sample_id=["id-a"],
        split_name=["train"],
        cohort_name=["easy_id"],
        source_dataset_name=["cifar10"],
        source_class_label=[1],
        candidate_class_mask=None,
    )
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    with pytest.raises(ValueError, match="batch_size < 2"):
        run_train_step(model, batch_input, optimizer, runtime_spec)


@torch.no_grad()
def _device_forward_smoke(backend_name: str):
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    runtime_spec = resolve_runtime(requested_backend=backend_name)
    model.to(runtime_spec.device)
    batch_on_device = batch_input.image.to(runtime_spec.device, dtype=runtime_spec.dtype)
    model_output = model(batch_on_device)
    assert torch.isfinite(model_output.class_mass).all()


def test_run_train_step_mps_smoke():
    if not HAS_MPS:
        return
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="mps")
    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)
    assert torch.isfinite(loss_breakdown.loss_total)


def test_run_train_step_rocm_smoke():
    if not HAS_ROCM:
        return
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="rocm")
    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)
    assert torch.isfinite(loss_breakdown.loss_total)


def test_run_train_step_cuda_smoke():
    if not HAS_CUDA:
        return
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cuda")
    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)
    assert torch.isfinite(loss_breakdown.loss_total)
