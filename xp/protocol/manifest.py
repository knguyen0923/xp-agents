from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xp.errors import ConfigMismatchError
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(cfg: dict, phase: Phase, out_dir, extra_versions: Optional[dict] = None) -> None:
    path = Path(out_dir) / phase.name / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text()).get("config_hash")
        if old != phase.config_hash:
            raise ConfigMismatchError(f"{path} was written under config {old}, current config is {phase.config_hash}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cfg": cfg,
        "config_hash": phase.config_hash,
        "git_commit": _git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **(extra_versions or {}),
        },
    }
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str)))
