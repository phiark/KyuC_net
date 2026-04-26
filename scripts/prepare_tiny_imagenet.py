#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "tiny_imagenet"
DEFAULT_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyImageNet for V0.6B directory-backed OOD loading.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default="", help="Expected archive SHA-256. Empty disables checksum validation.")
    parser.add_argument("--download", action="store_true", help="Download tiny-imagenet-200.zip when missing.")
    parser.add_argument("--extract", action="store_true", help="Extract tiny-imagenet-200.zip when the folder is missing.")
    return parser.parse_args()


def _download(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive_path)


def _sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_zip_member(root: Path, member_name: str) -> None:
    resolved_root = root.resolve()
    resolved_member = (root / member_name).resolve()
    try:
        resolved_member.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Archive member would extract outside output root: {member_name}") from exc


def _extract(archive_path: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _validate_zip_member(root, member.filename)
        archive.extractall(root)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    archive_path = root / "tiny-imagenet-200.zip"
    dataset_root = root / "tiny-imagenet-200"
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"

    if not dataset_root.exists():
        if args.download and not archive_path.exists():
            print(f"[tiny-imagenet] download {args.url} -> {archive_path}", flush=True)
            _download(args.url, archive_path)
        if args.extract:
            if not archive_path.exists():
                raise FileNotFoundError(f"TinyImageNet archive is missing: {archive_path}")
            if args.sha256:
                digest = _sha256_digest(archive_path)
                if digest.lower() != str(args.sha256).lower():
                    raise ValueError(f"TinyImageNet archive sha256 mismatch: {archive_path}")
            print(f"[tiny-imagenet] extract {archive_path} -> {root}", flush=True)
            _extract(archive_path, root)

    missing = [str(path) for path in (train_root, val_root) if not path.exists()]
    if missing:
        print("[tiny-imagenet] missing required directories:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        return 1

    print(f"[tiny-imagenet] ready root={dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
