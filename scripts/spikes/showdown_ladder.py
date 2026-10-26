"""THROWAWAY Showdown spike ladder runner: python showdown_ladder.py <format> <battles> [team_file]. Needs a local server and poke-env (see the plan)."""
from __future__ import annotations

import asyncio, sys, time, itertools
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from showdown_opponents import DamageHeuristicPlayer
import random as _r; SUF = "%d" % _r.randrange(10**6)
FMT = sys.argv[1]; N = int(sys.argv[2])
TEAM = {} if "random" in FMT else {"team": open(sys.argv[3]).read()}
TIERS = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "damage": DamageHeuristicPlayer}
def mk(tier, name):
    return TIERS[tier](account_configuration=AccountConfiguration(name + SUF, None), battle_format=FMT,
        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1, log_level=30, **TEAM)
async def main():
    for a, b in [("maxbp", "random"), ("damage", "maxbp"), ("damage", "random")]:
        pa, pb = mk(a, f"{a[:2]}{b[:2]}A"), mk(b, f"{b[:2]}{a[:2]}B")
        t = time.time()
        await asyncio.wait_for(pa.battle_against(pb, n_battles=N), 600)
        dt = time.time() - t
        w = pa.n_won_battles; n = pa.n_finished_battles
        p = w / n; se = (p * (1 - p) / n) ** 0.5
        print(f"{a} vs {b}: {w}/{n} = {p:.2f} (95% +-{1.96*se:.2f}), {dt:.0f}s, {n/dt:.2f} battles/s", flush=True)
asyncio.run(main())
