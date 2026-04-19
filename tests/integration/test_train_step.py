from __future__ import annotations

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
    assert torch.isfinite(loss_breakdown.loss_ambiguous)
    assert torch.isfinite(loss_breakdown.loss_total)


def test_run_train_step_cpu_smoke():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(model, batch_input, optimizer, runtime_spec)

    assert torch.isfinite(loss_breakdown.loss_total)
    assert loss_breakdown.optimizer_step_performed is True


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
