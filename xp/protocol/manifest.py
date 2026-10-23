from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xp.errors import ConfigMismatchError, ProtocolError
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(cfg: dict, phase: Phase, out_dir, extra_versions: Optional[dict] = None,
                   strict_code: bool = False, allow_code_change: bool = False) -> None:
    """Write the run manifest. Top-level keys describe the latest session; `sessions` is append-only."""
    path = Path(out_dir) / phase.name / "manifest.json"
    commit = _git_commit()
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **(extra_versions or {}),
    }
    sessions: list = []
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("config_hash") != phase.config_hash:
            raise ConfigMismatchError(
                f"{path} was written under config {old.get('config_hash')}, current config is {phase.config_hash}"
            )
        sessions = old.get("sessions") or [{"git_commit": old.get("git_commit"), "versions": old.get("versions")}]
        earlier = [s["git_commit"] for s in sessions if s.get("git_commit")]
        if strict_code and commit is not None and not allow_code_change:
            changed = [c for c in earlier if c != commit]
            if changed:
                raise ProtocolError(
                    f"code changed since this run started: earlier session(s) used commit {changed[0]}, "
                    f"current commit is {commit}. Results from different code must not be mixed silently; "
                    "rerun with --allow-code-change if the change is deliberate."
                )
    if not sessions or sessions[-1].get("git_commit") != commit:
        sessions.append({"git_commit": commit, "versions": versions})
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cfg": cfg,
        "config_hash": phase.config_hash,
        "git_commit": commit,
        "versions": versions,
        "sessions": sessions,
    }
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str)))
