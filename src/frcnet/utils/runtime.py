from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from frcnet.data import BatchInput

RequestedBackend = Literal["auto", "cpu", "mps", "cuda", "rocm"]
ResolvedBackend = Literal["cpu", "mps", "cuda", "rocm"]


@dataclass(slots=True)
class RuntimeSpec:
    requested_backend: RequestedBackend
    resolved_backend: ResolvedBackend
    device: torch.device
    dtype: torch.dtype
    amp_enabled: bool


def is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def is_rocm_available() -> bool:
    return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)


def is_cuda_available() -> bool:
    return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name != "float32":
        raise ValueError(f"Unsupported dtype for FRCNet 0.1: {dtype_name}")
    return torch.float32


def resolve_runtime(
    requested_backend: RequestedBackend = "auto",
    dtype: str = "float32",
    amp_enabled: bool = False,
) -> RuntimeSpec:
    resolved_backend: ResolvedBackend
    if requested_backend == "auto":
        if is_mps_available():
            resolved_backend = "mps"
        elif is_rocm_available():
            resolved_backend = "rocm"
        elif is_cuda_available():
            resolved_backend = "cuda"
        else:
            resolved_backend = "cpu"
    elif requested_backend == "mps":
        if not is_mps_available():
            raise RuntimeError("Requested backend 'mps' is not available.")
        resolved_backend = "mps"
    elif requested_backend == "rocm":
        if not is_rocm_available():
            raise RuntimeError("Requested backend 'rocm' is not available.")
        resolved_backend = "rocm"
    elif requested_backend == "cuda":
        if not is_cuda_available():
            raise RuntimeError("Requested backend 'cuda' is not available.")
        resolved_backend = "cuda"
    elif requested_backend == "cpu":
        resolved_backend = "cpu"
    else:
        raise ValueError(f"Unsupported backend request: {requested_backend}")

    if resolved_backend in {"cuda", "rocm"}:
        device = torch.device("cuda")
    elif resolved_backend == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return RuntimeSpec(
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        device=device,
        dtype=_resolve_dtype(dtype),
        amp_enabled=bool(amp_enabled and resolved_backend in {"cuda", "rocm"}),
    )


def resolve_pin_memory(pin_memory: str | bool, runtime_spec: RuntimeSpec) -> bool:
    if isinstance(pin_memory, bool):
        return pin_memory
    if pin_memory != "auto":
        raise ValueError(f"Unsupported pin_memory setting: {pin_memory}")
    return runtime_spec.resolved_backend in {"cuda", "rocm"}


def move_batch_to_device(batch_input: BatchInput, runtime_spec: RuntimeSpec) -> BatchInput:
    candidate_class_mask = batch_input.candidate_class_mask
    if candidate_class_mask is not None:
        candidate_class_mask = candidate_class_mask.to(runtime_spec.device)

    return BatchInput(
        image=batch_input.image.to(device=runtime_spec.device, dtype=runtime_spec.dtype),
        class_label=batch_input.class_label.to(runtime_spec.device),
        sample_id=list(batch_input.sample_id),
        split_name=list(batch_input.split_name),
        cohort_name=list(batch_input.cohort_name),
        source_dataset_name=list(batch_input.source_dataset_name),
        candidate_class_mask=candidate_class_mask,
    )

