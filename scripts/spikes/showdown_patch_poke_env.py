"""THROWAWAY: patch poke-env 0.10.0's bundled gen 9 data with the Champions-only species and moves.

Why: poke-env's Pokedex lacks 54 species (new Megas etc.) and 2 moves that the Showdown Champions
mod defines; a battle touching one raises KeyError in the message handler and hangs.
Steps (scratch dir, venv active, server checkout at ./pokemon-showdown built with node 22):
  1. python showdown_patch_poke_env.py missing   -> writes missing.json
  2. node showdown_export_patch.js               -> writes patch_data.json (needs champ_species.json /
     champ_moves.json from Dex.mod('champions'), see the plan's progress log)
  3. python showdown_patch_poke_env.py apply     -> merges into the venv's gen9 json (keeps *.orig)
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import poke_env
from poke_env.data import GenData

ROOT = os.path.join(os.path.dirname(poke_env.__file__), "data", "static")
PDEX = os.path.join(ROOT, "pokedex", "gen9pokedex.json")
MOVES = os.path.join(ROOT, "moves", "gen9moves.json")


def missing() -> None:
    g = GenData.from_gen(9)
    sp = set(json.load(open("champ_species.json")))
    mv = set(json.load(open("champ_moves.json")))
    json.dump({"species": sorted(sp - set(g.pokedex)), "moves": sorted(mv - set(g.moves))}, open("missing.json", "w"))


def apply() -> None:
    patch = json.load(open("patch_data.json"))
    for path, key in ((PDEX, "species"), (MOVES, "moves")):
        if not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        data = json.load(open(path + ".orig"))
        data.update(patch[key])
        json.dump(data, open(path, "w"), sort_keys=True)
        print(key, len(data))


if __name__ == "__main__":
    {"missing": missing, "apply": apply}[sys.argv[1]]()
