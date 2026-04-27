from __future__ import annotations

from pathlib import Path

import pytest

from frcnet.maintenance import (
    build_checkpoint_cleanup_plan,
    delete_checkpoint_candidates,
    should_retain_checkpoint,
    write_checkpoint_manifest,
)


def _write_file(path: Path, payload: bytes = b"checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_checkpoint_retention_rule_keeps_best_and_last() -> None:
    assert should_retain_checkpoint(Path("checkpoint_best.pt"))
    assert should_retain_checkpoint(Path("checkpoint_best_balanced.pt"))
    assert should_retain_checkpoint(Path("checkpoint_best_theory.safetensors"))
    assert should_retain_checkpoint(Path("checkpoint_best_near_ood_balanced.ckpt"))
    assert should_retain_checkpoint(Path("checkpoint_last.pt"))
    assert not should_retain_checkpoint(Path("checkpoint_epoch_001.pt"))


def test_checkpoint_cleanup_plan_dry_run_does_not_delete(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    best = _write_file(root / "run/checkpoints/checkpoint_best.pt")
    last = _write_file(root / "run/checkpoints/checkpoint_last.pt")
    epoch = _write_file(root / "run/checkpoints/checkpoint_epoch_001.pt")

    plan = build_checkpoint_cleanup_plan([root])

    assert {record.path for record in plan.retained_records} == {best, last}
    assert {record.path for record in plan.delete_records} == {epoch}
    assert best.exists()
    assert last.exists()
    assert epoch.exists()


def test_checkpoint_cleanup_execute_deletes_only_candidates(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    best = _write_file(root / "run/checkpoints/checkpoint_best.pt")
    epoch = _write_file(root / "run/checkpoints/checkpoint_epoch_001.pt")

    plan = build_checkpoint_cleanup_plan([root])
    deleted_count, deleted_size = delete_checkpoint_candidates(plan)

    assert deleted_count == 1
    assert deleted_size > 0
    assert best.exists()
    assert not epoch.exists()


def test_default_cleanup_roots_do_not_scan_unrelated_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifacts_epoch = _write_file(Path("artifacts/study/checkpoint_epoch_001.pt"))
    docs_epoch = _write_file(Path("docs/checkpoint_epoch_001.pt"))

    plan = build_checkpoint_cleanup_plan()

    assert {record.path for record in plan.delete_records} == {artifacts_epoch}
    assert docs_epoch.exists()


def test_cleanup_rejects_ambiguous_best_or_last_candidate(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _write_file(root / "run/checkpoints/model_best_shadow.pt")

    with pytest.raises(ValueError, match="best/last"):
        build_checkpoint_cleanup_plan([root])


def test_checkpoint_manifest_writer(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _write_file(root / "run/checkpoints/checkpoint_best.pt")
    _write_file(root / "run/checkpoints/checkpoint_epoch_001.pt")
    plan = build_checkpoint_cleanup_plan([root])
    manifest_path = write_checkpoint_manifest(plan.records, tmp_path / "manifest.csv")

    text = manifest_path.read_text(encoding="utf-8")
    assert "checkpoint_best.pt" in text
    assert "checkpoint_epoch_001.pt" in text
    assert "retain" in text
