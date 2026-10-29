from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, List, Optional

# Interest score of one battle, from the protocol log of one side. Four parts, each scaled to 0..1 and averaged:
#   closeness   winner ends with few Pokemon left (1 left = 1.0, all brought alive = 0.0)
#   comebacks   times the lead (alive count, agent minus opponent) changes sign, capped at LEAD_CAP
#   length      turns played, capped at TURN_CAP
#   mechanics   Mega Evolutions plus Terastallizations used by either side, capped at MECH_CAP
LEAD_CAP = 3
TURN_CAP = 15
MECH_CAP = 2

_LOG = re.compile(r'<script type="text/plain" class="battle-log-data">\s*(.*?)\s*</script>', re.S)
_NAME = re.compile(r"^(.*?) - (battle-.*)\.html$")


def read_log(page: str) -> List[str]:
    """Protocol lines of a poke-env replay page (`>battle-tag` is glued to the first event, so it is dropped)."""
    m = _LOG.search(page)
    if not m:
        raise ValueError("no battle log in the replay page")
    text = html.unescape(m.group(1))
    lines = text.split("\n")
    if lines and lines[0].startswith(">"):
        lines[0] = lines[0][lines[0].index("|"):] if "|" in lines[0] else ""
    return [ln for ln in lines if ln.startswith("|")]


def score_battle(lines: List[str], agent_name: str) -> Optional[Dict[str, float]]:
    """Score one battle for the side played by `agent_name`; None if it has no winner (unfinished)."""
    names: Dict[str, str] = {}
    brought, turns, mech, winner, tie = 4, 0, 0, None, False
    alive: Dict[str, int] = {}
    last_sign, changes = 0, 0
    for ln in lines:
        f = ln.split("|")
        kind = f[1] if len(f) > 1 else ""
        if kind == "player" and len(f) > 3 and f[2] in ("p1", "p2") and f[3]:
            names[f[2]] = f[3]
        elif kind == "teampreview" and len(f) > 2 and f[2].isdigit():
            brought = int(f[2])
        elif kind == "turn" and len(f) > 2 and f[2].isdigit():
            turns = int(f[2])
        elif kind in ("-mega", "-terastallize"):
            mech += 1
        elif kind == "faint" and len(f) > 2 and f[2][:2] in ("p1", "p2"):
            side = f[2][:2]
            alive[side] = alive.get(side, brought) - 1
            agent_side = "p1" if names.get("p1") == agent_name else "p2"
            other = "p2" if agent_side == "p1" else "p1"
            diff = alive.get(agent_side, brought) - alive.get(other, brought)
            sign = (diff > 0) - (diff < 0)
            if sign and last_sign and sign != last_sign:
                changes += 1
            if sign:
                last_sign = sign
        elif kind == "win" and len(f) > 2:
            winner = f[2]
        elif kind == "tie":
            tie = True
    if agent_name not in names.values():
        return None
    # poke-env saves the page before the final `|win|` line arrives, so the winner is usually inferred from the faint counts
    a1, a2 = alive.get("p1", brought), alive.get("p2", brought)
    if winner is None:
        if a1 > 0 and a2 > 0:
            return None  # unfinished
        if a1 == 0 and a2 == 0:
            winner = ""  # both sides ran out together: a draw
        else:
            winner = names.get("p1" if a1 > 0 else "p2", "")
    draw = winner == "" or tie
    win_side = "p1" if names.get("p1") == winner else "p2"
    left = 0 if draw else alive.get(win_side, brought)
    parts = {
        "closeness": 1.0 if draw else min(max((brought - left) / max(brought - 1, 1), 0.0), 1.0),
        "comebacks": min(changes, LEAD_CAP) / LEAD_CAP,
        "length": min(turns, TURN_CAP) / TURN_CAP,
        "mechanics": min(mech, MECH_CAP) / MECH_CAP,
    }
    parts["score"] = sum(parts.values()) / 4.0
    parts.update(turns=float(turns), won=0.5 if draw else float(winner == agent_name), left=float(left), lead_changes=float(changes), mechanics_used=float(mech))
    return parts


def rank_replays(folder, agent_suffix: str = "a", agent_prefix: Optional[str] = None) -> List[dict]:
    """Score every replay page in `folder` written for the agent's own account (usernames starting with `agent_prefix` if given, else ending in `agent_suffix`).

    Highest score first; ties break on more turns, then on the file name, so the order is deterministic.
    """
    rows = []
    for path in sorted(Path(folder).glob("*.html")):
        m = _NAME.match(path.name)
        if not m or not (m.group(1).startswith(agent_prefix) if agent_prefix else m.group(1).endswith(agent_suffix)):
            continue
        parts = score_battle(read_log(path.read_text(encoding="utf-8")), m.group(1))
        if parts is not None:
            rows.append({"file": path.name, **parts})
    rows.sort(key=lambda r: (-r["score"], -r["turns"], r["file"]))
    return rows
