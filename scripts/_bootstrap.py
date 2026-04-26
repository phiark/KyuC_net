from __future__ import annotations

import os
from pathlib import Path
import sys


def bootstrap_repo(*, configure_runtime_cache: bool = False) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if configure_runtime_cache:
        runtime_cache_root = repo_root / ".cache" / "runtime"
        runtime_cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache_root / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(repo_root / ".cache"))
    return repo_root
