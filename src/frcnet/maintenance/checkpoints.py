"""Checkpoint retention helpers for archive maintenance."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

MODEL_CHECKPOINT_EXTENSIONS = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})
DEFAULT_CHECKPOINT_CLEANUP_ROOTS = (Path("artifacts"), Path("records/experiments"))


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    size_bytes: int
    retain: bool

    @property
    def size_mib(self) -> float:
        return self.size_bytes / 1024**2


@dataclass(frozen=True)
class CheckpointCleanupPlan:
    records: tuple[CheckpointRecord, ...]

    @property
    def retained_records(self) -> tuple[CheckpointRecord, ...]:
        return tuple(record for record in self.records if record.retain)

    @property
    def delete_records(self) -> tuple[CheckpointRecord, ...]:
        return tuple(record for record in self.records if not record.retain)

    @property
    def total_size_bytes(self) -> int:
        return sum(record.size_bytes for record in self.records)

    @property
    def retained_size_bytes(self) -> int:
        return sum(record.size_bytes for record in self.retained_records)

    @property
    def delete_size_bytes(self) -> int:
        return sum(record.size_bytes for record in self.delete_records)


def is_model_checkpoint_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MODEL_CHECKPOINT_EXTENSIONS


def should_retain_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("checkpoint_best") or name.startswith("checkpoint_last")


def _normalise_roots(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    return tuple(Path(root) for root in roots)


def _iter_model_checkpoints(roots: Sequence[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if is_model_checkpoint_path(path))
    return tuple(sorted(paths))


def _validate_delete_candidates(records: Sequence[CheckpointRecord]) -> None:
    unsafe = [
        record.path
        for record in records
        if not record.retain and ("best" in record.path.name.lower() or "last" in record.path.name.lower())
    ]
    if unsafe:
        preview = ", ".join(str(path) for path in unsafe[:5])
        raise ValueError(f"Refusing to delete checkpoint names containing best/last: {preview}")


def build_checkpoint_cleanup_plan(
    roots: Iterable[str | Path] = DEFAULT_CHECKPOINT_CLEANUP_ROOTS,
) -> CheckpointCleanupPlan:
    root_paths = _normalise_roots(roots)
    records = tuple(
        CheckpointRecord(
            path=path,
            size_bytes=path.stat().st_size,
            retain=should_retain_checkpoint(path),
        )
        for path in _iter_model_checkpoints(root_paths)
    )
    _validate_delete_candidates(records)
    return CheckpointCleanupPlan(records=records)


def delete_checkpoint_candidates(plan: CheckpointCleanupPlan) -> tuple[int, int]:
    _validate_delete_candidates(plan.records)
    deleted_count = 0
    deleted_size_bytes = 0
    for record in plan.delete_records:
        if record.path.exists():
            deleted_size_bytes += record.path.stat().st_size
            record.path.unlink()
            deleted_count += 1
    return deleted_count, deleted_size_bytes


def write_checkpoint_manifest(records: Sequence[CheckpointRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "size_mib", "retain", "mtime_utc"])
        for record in records:
            mtime_utc = datetime.fromtimestamp(record.path.stat().st_mtime, timezone.utc).isoformat()
            writer.writerow(
                [
                    record.path.as_posix(),
                    record.size_bytes,
                    f"{record.size_mib:.2f}",
                    "true" if record.retain else "false",
                    mtime_utc,
                ]
            )
    return path
