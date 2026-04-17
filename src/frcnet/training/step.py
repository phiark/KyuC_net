from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from frcnet.data import BatchInput
from frcnet.training.losses import LossBreakdown, LossConfig, compute_total_loss
from frcnet.utils import RuntimeSpec, move_batch_to_device


def run_train_step(
    model: nn.Module,
    batch_input: BatchInput,
    optimizer: Optimizer,
    runtime_spec: RuntimeSpec,
    loss_config: LossConfig | dict | None = None,
) -> LossBreakdown:
    model.to(runtime_spec.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_on_device = move_batch_to_device(batch_input, runtime_spec)
    model_output = model(batch_on_device.image)
    loss_breakdown = compute_total_loss(model_output, batch_on_device, loss_config)
    loss_breakdown.loss_total.backward()
    optimizer.step()
    return loss_breakdown

