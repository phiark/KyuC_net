from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from frcnet.data import BatchInput
from frcnet.models import FRCNetModel, gradient_reverse
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
    assert torch.isfinite(loss_breakdown.loss_source_adv)
    assert torch.isfinite(loss_breakdown.loss_ood_supcon)
    assert torch.isfinite(loss_breakdown.loss_source_balanced_calibration)
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


def test_run_train_step_updates_ood_only_batch_when_source_loss_enabled():
    batch_input = BatchInput(
        image=torch.randn(2, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([-1, -1], dtype=torch.long),
        sample_id=["ood-a", "ood-b"],
        split_name=["train", "train"],
        cohort_name=["ood", "ood"],
        source_dataset_name=["svhn", "dtd"],
        source_class_label=[0, 1],
        source_domain_name=["svhn", "dtd"],
        source_domain_label=[1, 2],
        source_role=["seen_source_ood", "seen_source_ood"],
        candidate_class_mask=None,
    )
    model = FRCNetModel(num_classes=10, source_adversary_enabled=True, num_source_domains=7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(
        model,
        batch_input,
        optimizer,
        runtime_spec,
        {"source_adv_weight": 0.05},
    )

    assert loss_breakdown.num_trainable_samples == 2
    assert loss_breakdown.optimizer_step_performed is True
    assert loss_breakdown.loss_source_adv > 0.0


def test_run_train_step_skips_ood_only_supcon_without_in_domain_negatives():
    batch_input = BatchInput(
        image=torch.randn(2, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([-1, -1], dtype=torch.long),
        sample_id=["ood-a", "ood-b"],
        split_name=["train", "train"],
        cohort_name=["ood", "ood"],
        source_dataset_name=["svhn", "svhn"],
        source_class_label=[0, 1],
        source_domain_name=["svhn", "svhn"],
        source_domain_label=[1, 1],
        source_role=["seen_source_ood", "seen_source_ood"],
        candidate_class_mask=None,
    )
    model = FRCNetModel(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(
        model,
        batch_input,
        optimizer,
        runtime_spec,
        {"ood_supcon_weight": 0.10},
    )

    assert loss_breakdown.num_trainable_samples == 0
    assert loss_breakdown.optimizer_step_performed is False


def _source_invariant_batch() -> BatchInput:
    batch_size = 6
    candidate_class_mask = torch.zeros((batch_size, 10), dtype=torch.bool)
    candidate_class_mask[2, 2] = True
    candidate_class_mask[2, 3] = True
    return BatchInput(
        image=torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        class_label=torch.tensor([1, 4, -1, -1, -1, -1], dtype=torch.long),
        sample_id=[f"source-invariant-{index}" for index in range(batch_size)],
        split_name=["train"] * batch_size,
        cohort_name=["easy_id", "hard_id", "ambiguous_id", "unknown_supervision", "unknown_supervision", "ood"],
        source_dataset_name=["cifar10", "cifar10", "cifar10", "svhn", "dtd", "tiny_imagenet"],
        source_class_label=[1, 4, None, 7, 12, 21],
        source_domain_name=["cifar10", "cifar10", "cifar10", "svhn", "dtd", "tiny_imagenet"],
        source_domain_label=[0, 0, 0, 1, 2, 5],
        source_role=["in_domain", "in_domain", "in_domain", "seen_unknown_source", "seen_unknown_source", "ood"],
        candidate_class_mask=candidate_class_mask,
    )


def test_gradient_reverse_flips_gradient_sign():
    input_tensor = torch.tensor([2.0], requires_grad=True)
    loss = (gradient_reverse(input_tensor, lambda_value=0.25) ** 2).sum()

    loss.backward()

    torch.testing.assert_close(input_tensor.grad, torch.tensor([-1.0]))


def test_source_head_disabled_preserves_default_output_and_checkpoint_loading(tmp_path):
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored = FRCNetModel(num_classes=10)
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model_output = restored(batch_input.image)

    assert model_output.source_logits is None


def test_source_adversary_loss_requires_logits_and_source_domain_labels():
    batch_input = _source_invariant_batch()
    model_output_without_source_head = FRCNetModel(num_classes=10)(batch_input.image)

    with pytest.raises(ValueError, match="source_logits"):
        compute_total_loss(model_output_without_source_head, batch_input, {"source_adv_weight": 0.05})

    model_output_with_source_head = FRCNetModel(
        num_classes=10,
        source_adversary_enabled=True,
        num_source_domains=7,
    )(batch_input.image)
    batch_without_labels = replace(batch_input, source_domain_label=None)
    with pytest.raises(ValueError, match="source_domain_label"):
        compute_total_loss(model_output_with_source_head, batch_without_labels, {"source_adv_weight": 0.05})


def test_source_invariant_losses_are_finite_when_enabled():
    batch_input = _source_invariant_batch()
    model_output = FRCNetModel(
        num_classes=10,
        source_adversary_enabled=True,
        num_source_domains=7,
    )(batch_input.image)

    loss_breakdown = compute_total_loss(
        model_output,
        batch_input,
        {
            "source_adv_weight": 0.05,
            "ood_supcon_weight": 0.10,
            "source_balanced_calibration_weight": 0.05,
        },
    )

    assert torch.isfinite(loss_breakdown.loss_source_adv)
    assert torch.isfinite(loss_breakdown.loss_ood_supcon)
    assert torch.isfinite(loss_breakdown.loss_source_balanced_calibration)
    assert loss_breakdown.loss_source_adv > 0.0


def test_ood_only_source_loss_batch_performs_optimizer_step():
    batch_input = BatchInput(
        image=torch.randn(4, 3, 32, 32, dtype=torch.float32),
        class_label=torch.full((4,), -1, dtype=torch.long),
        sample_id=[f"ood-{index}" for index in range(4)],
        split_name=["train"] * 4,
        cohort_name=["ood"] * 4,
        source_dataset_name=["svhn", "dtd", "svhn", "dtd"],
        source_domain_name=["svhn", "dtd", "svhn", "dtd"],
        source_domain_label=[1, 2, 1, 2],
        source_role=["seen_source_ood"] * 4,
        source_partition_name=["unit"] * 4,
    )
    model = FRCNetModel(num_classes=10, source_adversary_enabled=True, num_source_domains=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime_spec = resolve_runtime(requested_backend="cpu")

    loss_breakdown = run_train_step(
        model,
        batch_input,
        optimizer,
        runtime_spec,
        {"source_adv_weight": 0.05},
    )

    assert loss_breakdown.num_trainable_samples == 4
    assert loss_breakdown.optimizer_step_performed is True


def test_ood_supcon_returns_connected_zero_without_cross_source_positive():
    batch_input = replace(
        _source_invariant_batch(),
        cohort_name=["easy_id", "hard_id", "ambiguous_id", "unknown_supervision", "unknown_supervision", "ood"],
        source_dataset_name=["cifar10", "cifar10", "cifar10", "svhn", "svhn", "svhn"],
        source_domain_name=["cifar10", "cifar10", "cifar10", "svhn", "svhn", "svhn"],
        source_domain_label=[0, 0, 0, 1, 1, 1],
    )
    model_output = FRCNetModel(num_classes=10)(batch_input.image)

    loss_breakdown = compute_total_loss(model_output, batch_input, {"ood_supcon_weight": 0.10})

    assert torch.isfinite(loss_breakdown.loss_ood_supcon)
    assert loss_breakdown.loss_ood_supcon.item() == pytest.approx(0.0)
    assert loss_breakdown.loss_ood_supcon.requires_grad


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
