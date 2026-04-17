from __future__ import annotations

import pytest
import torch

from frcnet.utils.runtime import resolve_pin_memory, resolve_runtime


def test_resolve_runtime_prefers_mps(monkeypatch):
    monkeypatch.setattr("frcnet.utils.runtime.is_mps_available", lambda: True)
    monkeypatch.setattr("frcnet.utils.runtime.is_rocm_available", lambda: True)
    monkeypatch.setattr("frcnet.utils.runtime.is_cuda_available", lambda: True)

    runtime_spec = resolve_runtime()

    assert runtime_spec.resolved_backend == "mps"
    assert runtime_spec.device == torch.device("mps")


def test_resolve_runtime_detects_rocm(monkeypatch):
    monkeypatch.setattr("frcnet.utils.runtime.is_mps_available", lambda: False)
    monkeypatch.setattr("frcnet.utils.runtime.is_rocm_available", lambda: True)
    monkeypatch.setattr("frcnet.utils.runtime.is_cuda_available", lambda: True)

    runtime_spec = resolve_runtime()

    assert runtime_spec.resolved_backend == "rocm"
    assert runtime_spec.device == torch.device("cuda")


def test_resolve_runtime_detects_cuda(monkeypatch):
    monkeypatch.setattr("frcnet.utils.runtime.is_mps_available", lambda: False)
    monkeypatch.setattr("frcnet.utils.runtime.is_rocm_available", lambda: False)
    monkeypatch.setattr("frcnet.utils.runtime.is_cuda_available", lambda: True)

    runtime_spec = resolve_runtime()

    assert runtime_spec.resolved_backend == "cuda"
    assert runtime_spec.device == torch.device("cuda")


def test_resolve_runtime_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("frcnet.utils.runtime.is_mps_available", lambda: False)
    monkeypatch.setattr("frcnet.utils.runtime.is_rocm_available", lambda: False)
    monkeypatch.setattr("frcnet.utils.runtime.is_cuda_available", lambda: False)

    runtime_spec = resolve_runtime()

    assert runtime_spec.resolved_backend == "cpu"
    assert runtime_spec.device == torch.device("cpu")


def test_resolve_runtime_rejects_unavailable_explicit_backend(monkeypatch):
    monkeypatch.setattr("frcnet.utils.runtime.is_cuda_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_runtime(requested_backend="cuda")


def test_resolve_pin_memory_auto():
    assert resolve_pin_memory("auto", resolve_runtime(requested_backend="cpu")) is False

