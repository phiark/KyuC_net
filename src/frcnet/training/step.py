from __future__ import annotations

import torch.nn
import torch.nn as nn
from torch.optim import Optimizer

from frcnet.data import BatchInput
from frcnet.training.losses import LossBreakdown, LossConfig, compute_total_loss
from frcnet.utils import RuntimeSpec, move_batch_to_device


def _model_requires_batch_size_at_least_two(model: nn.Module) -> bool:
    batchnorm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    return any(isinstance(module, batchnorm_types) for module in model.modules())


def run_train_step(
    model: nn.Module,
    batch_input: BatchInput,
    optimizer: Optimizer,
    runtime_spec: RuntimeSpec,
    loss_config: LossConfig | dict | None = None,
) -> LossBreakdown:
    if batch_input.batch_size < 2 and _model_requires_batch_size_at_least_two(model):
        raise ValueError(
            "Training batches with batch_size < 2 are unsupported for BatchNorm-backed models. "
            "Use drop_last=True or increase the batch size."
        )

    model.to(runtime_spec.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_on_device = move_batch_to_device(batch_input, runtime_spec)
    model_output = model(batch_on_device.image)
    loss_breakdown = compute_total_loss(model_output, batch_on_device, loss_config)
    if loss_breakdown.num_trainable_samples > 0:
        loss_breakdown.loss_total.backward()
        optimizer.step()
        loss_breakdown.optimizer_step_performed = True
    return loss_breakdown
