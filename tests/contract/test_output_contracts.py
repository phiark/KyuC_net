from __future__ import annotations

import torch

from frcnet.models import FRCNetModel
from frcnet.utils import completion_score, content_entropy
from tests.conftest import build_synthetic_batch


def test_model_output_contract_fields_and_invariants():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)

    model_output = model(batch_input.image)

    assert model_output.backbone_feature.shape[0] == batch_input.batch_size
    assert model_output.resolution_logit.shape == (batch_input.batch_size,)
    assert model_output.resolution_ratio.shape == (batch_input.batch_size,)
    assert model_output.content_logits.shape == (batch_input.batch_size, 10)
    assert model_output.content_distribution.shape == (batch_input.batch_size, 10)
    assert model_output.class_mass.shape == (batch_input.batch_size, 10)
    assert model_output.unknown_mass.shape == (batch_input.batch_size,)

    torch.testing.assert_close(
        model_output.class_mass.sum(dim=1) + model_output.unknown_mass,
        torch.ones(batch_input.batch_size),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        model_output.unknown_mass,
        1.0 - model_output.resolution_ratio,
        atol=1e-5,
        rtol=1e-5,
    )


def test_scoring_helpers_are_stable_on_model_output():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    entropy = content_entropy(model_output.content_distribution)
    score = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.1)

    assert entropy.shape == (batch_input.batch_size,)
    assert score.shape == (batch_input.batch_size,)
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(score).all()

