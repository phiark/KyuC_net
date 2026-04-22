from __future__ import annotations

import torch
import pytest

from frcnet.utils import (
    completion_score,
    content_entropy,
    resolution_entropy,
    resolution_weighted_content_entropy,
    ternary_entropy_from_masses,
)


def test_content_entropy_returns_expected_shape():
    distribution = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)

    entropy = content_entropy(distribution)

    assert entropy.shape == (2,)
    assert torch.all(entropy >= 0)


def test_completion_score_matches_top1_plus_unknown():
    class_mass = torch.tensor([[0.4, 0.3, 0.1], [0.2, 0.5, 0.0]], dtype=torch.float32)
    unknown_mass = torch.tensor([0.2, 0.3], dtype=torch.float32)

    score = completion_score(class_mass, unknown_mass, beta=0.5)

    expected = torch.tensor([0.5, 0.65], dtype=torch.float32)
    torch.testing.assert_close(score, expected)


def test_completion_score_rejects_invalid_beta():
    with pytest.raises(ValueError):
        completion_score(torch.ones(1, 2), torch.zeros(1), beta=1.1)


def test_resolution_weighted_content_entropy_multiplies_inputs():
    resolution_ratio = torch.tensor([0.2, 0.8], dtype=torch.float32)
    entropy = torch.tensor([0.5, 1.5], dtype=torch.float32)

    weighted_entropy = resolution_weighted_content_entropy(resolution_ratio, entropy)

    expected = torch.tensor([0.1, 1.2], dtype=torch.float32)
    torch.testing.assert_close(weighted_entropy, expected)


def test_resolution_entropy_returns_binary_entropy():
    resolution_ratio = torch.tensor([0.5, 1.0], dtype=torch.float32)

    entropy = resolution_entropy(resolution_ratio)

    assert entropy.shape == (2,)
    assert entropy[0] > entropy[1]


def test_ternary_entropy_accepts_truth_false_unknown_masses():
    truth_mass = torch.tensor([0.2], dtype=torch.float32)
    false_mass = torch.tensor([0.3], dtype=torch.float32)
    unknown_mass = torch.tensor([0.5], dtype=torch.float32)

    entropy = ternary_entropy_from_masses(truth_mass, false_mass, unknown_mass)

    assert entropy.shape == (1,)
    assert torch.isfinite(entropy).all()
