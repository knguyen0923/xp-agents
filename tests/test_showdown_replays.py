from __future__ import annotations

from xp.showdown.replays import rank_replays, read_log, score_battle

PAGE = """<html><script type="text/plain" class="battle-log-data">
    {log}
    </script></html>"""


def page(lines, tag="battle-gen9championsvgc2026regmc-1"):
    return PAGE.format(log=f">{tag}" + "\n".join(lines))


# agent is p1 ("vxa"); four brought per side
BASE = ["|player|p1|vxa|1|", "|player|p2|vxb|2|", "|teampreview|4", "|turn|1"]


def close_comeback():
    """Agent falls behind 2 to 4 in alive count, comes back to lead, wins with one left; a Mega is used."""
    return BASE + ["|-mega|p1a: A|A|Venusaurite", "|faint|p1a: A", "|faint|p1b: B", "|turn|5", "|faint|p2a: X", "|faint|p2b: Y",
                   "|faint|p2a: Z", "|turn|9", "|faint|p1a: C", "|win|vxa"]


def stomp():
    return BASE + ["|turn|2", "|faint|p2a: X", "|faint|p2b: Y", "|faint|p2a: Z", "|win|vxa"]


def test_read_log_drops_tag_and_unescapes():
    lines = read_log(page(["|player|p1|a&amp;b|1|", "|turn|1"]))
    assert lines == ["|player|p1|a&b|1|", "|turn|1"]


def test_close_comeback_scores_higher_than_stomp():
    a, b = score_battle(close_comeback(), "vxa"), score_battle(stomp(), "vxa")
    assert a["score"] > b["score"]
    assert a["left"] == 1 and b["left"] == 4  # stomp: the agent lost nothing
    assert a["lead_changes"] == 1 and b["lead_changes"] == 0
    assert a["mechanics_used"] == 1 and a["won"] == 1.0


def test_winner_survivors_drive_closeness():
    a, b = score_battle(close_comeback(), "vxa"), score_battle(stomp(), "vxa")
    assert a["closeness"] == 1.0 and b["closeness"] == 0.0


def test_unfinished_battle_has_no_score():
    assert score_battle(BASE + ["|faint|p1a: A"], "vxa") is None


def test_loss_counts_from_the_winner_side():
    lines = BASE + ["|faint|p1a: A", "|faint|p1b: B", "|faint|p1a: C", "|win|vxb"]
    s = score_battle(lines, "vxa")
    assert s["won"] == 0.0 and s["left"] == 4.0


def test_rank_skips_the_opponents_account_and_orders_deterministically(tmp_path):
    for name, lines in (("vxa - battle-g-1.html", close_comeback()), ("vxb - battle-g-1.html", close_comeback()),
                        ("vxa - battle-g-2.html", stomp())):
        (tmp_path / name).write_text(page(lines), encoding="utf-8")
    rows = rank_replays(tmp_path)
    assert [r["file"] for r in rows] == ["vxa - battle-g-1.html", "vxa - battle-g-2.html"]
