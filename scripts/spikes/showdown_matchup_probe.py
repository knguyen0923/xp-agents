"""Throwaway probe: what does poke-env know about a real doubles battle, and can calculate_damage run on it?"""
from __future__ import annotations

import time

from poke_env import AccountConfiguration
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.environment import SingleAgentWrapper

from xp.showdown.env import XPDoublesEnv
from xp.showdown.opponents import make_opponent
from xp.showdown.teams import PoolTeambuilder, load_pool

FORMAT = "gen9championsvgc2026regmc"


def main() -> None:
    pool = load_pool("configs/showdown_teams/tune")
    env = XPDoublesEnv(battle_format=FORMAT, team=PoolTeambuilder(pool, 1), strict=False, log_level=40,
                       account_configuration1=AccountConfiguration("ProbeAgent3", None),
                       account_configuration2=AccountConfiguration("ProbeOpp3", None))
    wrapped = SingleAgentWrapper(env, make_opponent("damage", FORMAT))
    wrapped.reset()
    b = env.battle1
    print("player_role", b.player_role, "opponent_role", b.opponent_role)
    print("team keys", list(b.team), "opp team keys", list(b.opponent_team))
    for label, mons in (("own", b.active_pokemon), ("opp", b.opponent_active_pokemon)):
        for m in mons:
            if m is None:
                continue
            print(label, m.species, "ability", m.ability, "item", m.item, "stats", m.stats, "hp", m.current_hp, m.max_hp,
                  "moves", list(m.moves))
    # Can calculate_damage run for own -> opp and opp -> own? How long does one call take?
    own = [m for m in b.active_pokemon if m is not None]
    opp = [m for m in b.opponent_active_pokemon if m is not None]

    def ident(mon):
        for team in (b.team, b.opponent_team):
            for key, value in team.items():
                if value is mon:
                    return key
        raise KeyError(mon.species)

    for a, d in ((own[0], opp[0]), (opp[0], own[0])):
        for move in list(a.moves.values())[:4]:
            t0 = time.perf_counter()
            try:
                out = calculate_damage(ident(a), ident(d), move, b)
            except Exception as e:  # report, never hide
                out = f"{type(e).__name__}: {e}"
            print(a.species, "->", d.species, move.id, "calc", out, "mult", d.damage_multiplier(move),
                  f"{(time.perf_counter() - t0) * 1000:.2f} ms")
    # Control: own -> own (defender stats known) to separate "calculator broken" from "foe stats missing".
    if len(own) == 2:
        for move in list(own[0].moves.values())[:4]:
            t0 = time.perf_counter()
            try:
                out = calculate_damage(ident(own[0]), ident(own[1]), move, b)
            except Exception as e:
                out = f"{type(e).__name__}: {e}"
            print("CONTROL", own[0].species, "->", own[1].species, move.id, "calc", out,
                  f"{(time.perf_counter() - t0) * 1000:.2f} ms")
    wrapped.close()


if __name__ == "__main__":
    main()
