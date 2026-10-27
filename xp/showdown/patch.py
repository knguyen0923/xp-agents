"""Merge the Champions-only species and moves into poke-env 0.10.0's bundled gen 9 data.

Why: poke-env's Pokedex lacks 54 species (new Megas and others) and 2 moves that the Showdown Champions mod defines.
A battle that touches one raises KeyError inside poke-env's message handler and the battle HANGS with no error.
The data in `champions_patch.json` was exported from the server's Champions mod (`Dex.mod('champions')`).
Idempotent; originals are kept as `*.orig`. Run: python -m xp.showdown.patch
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

PATCH = os.path.join(os.path.dirname(__file__), "champions_patch.json")


def _paths() -> Dict[str, str]:
    import poke_env

    root = os.path.join(os.path.dirname(poke_env.__file__), "data", "static")
    return {"species": os.path.join(root, "pokedex", "gen9pokedex.json"), "moves": os.path.join(root, "moves", "gen9moves.json")}


def missing() -> Dict[str, List[str]]:
    """Patch keys not yet present in the installed poke-env data (empty lists once patched)."""
    patch = json.load(open(PATCH))
    out = {}
    for kind, path in _paths().items():
        have = json.load(open(path))
        out[kind] = sorted(set(patch[kind]) - set(have))
    return out


def apply() -> Dict[str, int]:
    """Merge the patch into the installed data files; returns the number of entries added per kind."""
    patch = json.load(open(PATCH))
    added = {}
    for kind, path in _paths().items():
        if not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        data = json.load(open(path + ".orig"))
        before = len(data)
        data.update(patch[kind])
        with open(path, "w") as f:
            json.dump(data, f, sort_keys=True)
        added[kind] = len(data) - before
    return added


if __name__ == "__main__":
    print("added", apply(), "still missing", missing())
