from __future__ import annotations

from importlib import metadata as package_metadata
from pathlib import Path
import tomllib

import frcnet
from frcnet._version import __version__


def test_version_uses_single_source() -> None:
    assert frcnet.__version__ == "0.6.0"
    assert __version__ == "0.6.0"

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == (
        "frcnet._version.__version__"
    )


def test_archive_status_declares_current_version() -> None:
    archive_status = Path("docs/governance/project_archive_status.md").read_text(encoding="utf-8")
    archive_closure = Path("records/reviews/2026-04-27_review_v0_6c_archive_closure.md").read_text(
        encoding="utf-8"
    )

    assert "- version: `0.6.0`" in archive_status
    assert "FRCNet 0.6.0" in archive_closure


def test_installed_package_metadata_matches_when_available() -> None:
    try:
        installed_version = package_metadata.version("frcnet")
    except package_metadata.PackageNotFoundError:
        return

    assert installed_version == __version__
