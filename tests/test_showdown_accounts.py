from __future__ import annotations

from xp.showdown.vec_env import account_names


def test_account_names_are_readable_short_alphanumeric_and_distinct():
    names = [n for i in range(8) for n in account_names(("Kaiser", "Heuristic"), i, "472")]
    assert names[:2] == ["Kaiser0x472", "Heuristic0x472"]
    assert len(set(names)) == len(names)
    assert all(n.isalnum() and len(n) <= 18 for n in names)


def test_account_names_truncate_long_labels():
    agent, opp = account_names(("A" * 30, "B" * 30), 7, "472")
    assert len(agent) == 18 and len(opp) == 18
