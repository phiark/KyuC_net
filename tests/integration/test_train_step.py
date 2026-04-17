from __future__ import annotations

import torch

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

