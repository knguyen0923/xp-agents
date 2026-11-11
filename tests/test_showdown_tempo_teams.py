from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from xp.showdown.teams import assert_disjoint, load_pool, validate_team

ROOT = Path(__file__).resolve().parent.parent
TEMPO = ROOT / "configs" / "showdown_teams_tempo"
FORMAT = "gen9championsvgc2026regmc"
KIND = {"Trick Room": "trick_room", "Tailwind": "tailwind", "Reflect": "screen", "Light Screen": "screen", "Aurora Veil": "screen"}


def kinds(text: str) -> set:
    return {KIND[m] for m in re.findall(r"^- (.+?)\s*$", text, re.M) if m in KIND}


@pytest.mark.parametrize("stage", ["tune", "eval"])
def test_every_team_carries_a_setter_and_each_pool_covers_all_three_kinds(stage):
    pool = load_pool(TEMPO / stage)
    assert len(pool) == 3
    for name, text in pool.items():
        assert kinds(text), f"{name} has no Trick Room, Tailwind or screen move"
    assert set().union(*(kinds(t) for t in pool.values())) == {"trick_room", "tailwind", "screen"}


def test_tune_and_eval_pools_are_disjoint():
    assert_disjoint(load_pool(TEMPO / "tune"), load_pool(TEMPO / "eval"))


def test_the_original_pools_are_untouched():
    from xp.showdown.teams import pool_digest

    assert pool_digest(load_pool(ROOT / "configs" / "showdown_teams" / "tune")) == "4ebb9de5ef49"
    assert pool_digest(load_pool(ROOT / "configs" / "showdown_teams" / "eval")) == "bcc8a297c758"


@pytest.mark.showdown
def test_every_tempo_team_validates_on_the_server_checkout():
    showdown_dir = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))
    if not showdown_dir.exists():
        pytest.skip(f"no Showdown checkout at {showdown_dir} (set XP_SHOWDOWN_DIR)")
    for stage in ("tune", "eval"):
        for text in load_pool(TEMPO / stage).values():
            validate_team(text, FORMAT, showdown_dir, node=os.environ.get("XP_NODE", "node"))
