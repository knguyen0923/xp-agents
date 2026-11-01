from __future__ import annotations

from xp.showdown.replays import rank_replays, read_log, score_battle, select_top

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
                   "|faint|p2a: Z", "|turn|9", "|faint|p1a: C", "|faint|p2b: W", "|win|vxa"]


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


def test_winner_is_inferred_when_the_win_line_is_missing():
    s = score_battle([ln for ln in close_comeback() if not ln.startswith("|win|")], "vxa")
    assert s["won"] == 1.0 and s["left"] == 1.0
    lost = BASE + ["|faint|p1a: A", "|faint|p1b: B", "|faint|p1a: C", "|faint|p1b: D"]
    assert score_battle(lost, "vxa")["won"] == 0.0


def test_both_sides_out_together_is_a_draw():
    lines = BASE + ["|faint|p1a: A", "|faint|p1b: B", "|faint|p1a: C", "|faint|p1b: D",
                    "|faint|p2a: W", "|faint|p2b: X", "|faint|p2a: Y", "|faint|p2b: Z"]
    s = score_battle(lines, "vxa")
    assert s["won"] == 0.5 and s["closeness"] == 1.0


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


def test_rank_by_agent_name_prefix(tmp_path):
    def renamed(agent, opp):
        return [ln.replace("vxa", agent).replace("vxb", opp) for ln in close_comeback()]

    for name, lines in (("Kaiser3x472 - battle-g-1.html", renamed("Kaiser3x472", "Heuristic3x472")),
                        ("Heuristic3x472 - battle-g-1.html", renamed("Kaiser3x472", "Heuristic3x472")),
                        ("Kaiser4x472 - battle-g-2.html", renamed("Kaiser4x472", "Heuristic4x472"))):
        (tmp_path / name).write_text(page(lines), encoding="utf-8")
    rows = rank_replays(tmp_path, agent_prefix="Kaiser")
    assert sorted(r["file"] for r in rows) == ["Kaiser3x472 - battle-g-1.html", "Kaiser4x472 - battle-g-2.html"]


def test_select_top_keeps_best_within_the_turn_cap():
    rows = [{"file": "long", "turns": 30.0}, {"file": "mid", "turns": 9.0}, {"file": "short", "turns": 2.0}]
    assert [r["file"] for r in select_top(rows, 2)] == ["long", "mid"]
    assert [r["file"] for r in select_top(rows, 2, max_turns=15)] == ["mid", "short"]
    assert select_top(rows, 2, max_turns=1) == []


def test_select_top_wins_only_within_a_turn_range():
    rows = [{"file": "loss", "turns": 8.0, "won": 0.0}, {"file": "draw", "turns": 8.0, "won": 0.5}, {"file": "long", "turns": 14.0, "won": 1.0},
            {"file": "ok1", "turns": 9.0, "won": 1.0}, {"file": "short", "turns": 3.0, "won": 1.0}, {"file": "ok2", "turns": 5.0, "won": 1.0}]
    assert [r["file"] for r in select_top(rows, 5, min_turns=5, max_turns=12, wins_only=True)] == ["ok1", "ok2"]
    assert [r["file"] for r in select_top(rows, 1, min_turns=5, max_turns=12, wins_only=True)] == ["ok1"]
    assert [r["file"] for r in select_top(rows, 9, min_turns=5)] == ["loss", "draw", "long", "ok1", "ok2"]
