from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path
from typing import Dict, List

from poke_env.teambuilder import Teambuilder

from xp.errors import ProtocolError


def normalise(text: str) -> str:
    """Team identity ignores trailing spaces and blank-line differences."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def load_pool(directory) -> Dict[str, str]:
    """name (file stem) -> export text, for every *.txt in `directory`, sorted by name."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise ProtocolError(f"no team files in {directory}")
    return {p.stem: p.read_text() for p in paths}


def pool_digest(pool: Dict[str, str]) -> str:
    """Short hash of a pool's names and normalised team texts; a config pins it so edited team files cannot reuse old caches."""
    h = hashlib.sha256()
    for name in sorted(pool):
        h.update(name.encode() + b"\0" + normalise(pool[name]).encode() + b"\0")
    return h.hexdigest()[:12]


def assert_disjoint(tune: Dict[str, str], evals: Dict[str, str]) -> None:
    """Tune and eval pools must share no team (same text up to whitespace) and no file name."""
    same_name = set(tune) & set(evals)
    if same_name:
        raise ProtocolError(f"tune and eval pools share team names {sorted(same_name)}")
    seen = {normalise(t) for t in tune.values()}
    dup = [n for n, t in evals.items() if normalise(t) in seen]
    if dup:
        raise ProtocolError(f"eval teams {dup} also appear in the tune pool")


def validate_team(text: str, format_id: str, showdown_dir, node: str = "node") -> None:
    """Run the Showdown validator on one team. `poke-env` hangs silently on a rejected team, so call this first."""
    proc = subprocess.run([node, "pokemon-showdown", "validate-team", format_id], input=text, cwd=str(showdown_dir),
                          capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or out:
        raise ProtocolError(f"team rejected for {format_id}: {out or 'exit code ' + str(proc.returncode)}")


class PoolTeambuilder(Teambuilder):
    """Draws one team per battle, only from its own pool, with a seeded RNG. `drawn` logs the names."""

    def __init__(self, teams: Dict[str, str], seed: int):
        if not teams:
            raise ProtocolError("PoolTeambuilder needs at least one team")
        self.teams = dict(teams)
        self.names: List[str] = sorted(self.teams)
        self.drawn: List[str] = []
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        name = self._rng.choice(self.names)
        self.drawn.append(name)
        return self.join_team(self.parse_showdown_team(self.teams[name]))
