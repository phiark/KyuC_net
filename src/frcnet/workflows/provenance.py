from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(input_path: str | Path) -> str:
    path = Path(input_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_stage_provenance(
    *,
    stage_name: str,
    input_files: Iterable[str | Path] = (),
    input_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    file_payload = []
    for input_file in input_files:
        if input_file in {None, ""}:  # type: ignore[comparison-overlap]
            continue
        path = Path(input_file)
        if not path.exists():
            file_payload.append({"path": str(path), "exists": False, "sha256": ""})
            continue
        file_payload.append({"path": str(path), "exists": True, "sha256": file_sha256(path)})
    payload = {
        "stage_name": stage_name,
        "input_files": file_payload,
        "input_values": dict(input_values or {}),
    }
    payload["provenance_hash"] = stable_hash(payload)
    return payload


def write_stage_provenance(
    output_dir: str | Path,
    provenance_payload: Mapping[str, Any],
    *,
    filename: str = "stage_provenance.json",
) -> Path:
    output = Path(output_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(provenance_payload), indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_stage_provenance(
    output_dir: str | Path,
    *,
    filename: str = "stage_provenance.json",
) -> dict[str, Any] | None:
    path = Path(output_dir) / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stage_provenance(
    output_dir: str | Path,
    expected_provenance: Mapping[str, Any],
    *,
    resume_policy: str,
    stage_label: str,
    filename: str = "stage_provenance.json",
) -> bool:
    observed = read_stage_provenance(output_dir, filename=filename)
    expected_hash = str(expected_provenance["provenance_hash"])
    observed_hash = "" if observed is None else str(observed.get("provenance_hash", ""))
    if observed_hash == expected_hash:
        return True
    if resume_policy == "rebuild_stale":
        return False
    if resume_policy == "fail_on_stale":
        if observed is None:
            raise ValueError(f"{stage_label} outputs exist but have no stage provenance.")
        raise ValueError(
            f"{stage_label} outputs are stale: expected provenance {expected_hash}, observed {observed_hash}."
        )
    if resume_policy == "legacy_snapshot":
        return True
    raise ValueError("study.resume_policy must be one of fail_on_stale, rebuild_stale, or legacy_snapshot.")
