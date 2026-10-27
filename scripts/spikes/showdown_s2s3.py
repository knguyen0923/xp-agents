"""THROWAWAY S2 (ladder on the real Reg M-C teams, teams swapped between sides) and S3 (throughput). Needs a local server and the team files in ~/xp-showdown-spike/teams (see the plan)."""
from __future__ import annotations
import asyncio, sys, time, random, subprocess
sys.path.insert(0, "/Users/knguyen/VSC/xp-agents/scripts/spikes")
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from showdown_opponents import DamageHeuristicPlayer
FMT = "gen9championsvgc2026regmc"; SUF = "%d" % random.randrange(10**5)
T = ["tune1", "tune2", "tune3", "eval1", "eval2", "eval3"]
TEAM = {t: open(f"/Users/knguyen/xp-showdown-spike/teams/{t}.txt").read() for t in T}
TIERS = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "damage": DamageHeuristicPlayer}
PID = subprocess.check_output(["lsof", "-tiTCP:8000", "-sTCP:LISTEN"]).split()[0].decode()
_n = 0
def mk(tier, team, conc=1):
    global _n; _n += 1
    return TIERS[tier](account_configuration=AccountConfiguration(f"p{_n}x{SUF}", None), battle_format=FMT,
        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=conc, log_level=30, team=TEAM[team])
def cpu():  # server cumulative CPU seconds
    s = subprocess.check_output(["ps", "-p", PID, "-o", "cputime="]).decode().strip()
    parts = [float(x) for x in s.replace("-", ":").split(":")]; v = 0.0
    for x in parts: v = v * 60 + x
    return v
async def match(ta, tb, tma, tmb, n, conc=1):
    a, b = mk(ta, tma, conc), mk(tb, tmb, conc)
    await asyncio.wait_for(a.battle_against(b, n_battles=n), 300)
    return a.n_won_battles, a.n_finished_battles, sum(bt.turn for bt in a.battles.values())
async def s2(n_per_cell):
    for ta, tb in [("maxbp", "random"), ("damage", "maxbp"), ("damage", "random")]:
        w = f = turns = 0; t0 = time.time()
        for i, t in enumerate(T):
            u = T[(i + 1) % 6]
            for x, y in ((t, u), (u, t)):   # both team assignments
                ww, ff, tt = await match(ta, tb, x, y, n_per_cell); w += ww; f += ff; turns += tt
        p = w / f; se = (p * (1 - p) / f) ** 0.5
        print(f"S2 {ta} vs {tb}: {w}/{f} = {p:.3f} (95% +-{1.96*se:.3f}), mean turns {turns/f:.1f}, {time.time()-t0:.0f}s", flush=True)
async def s3(n, conc):
    c0, t0 = cpu(), time.time()
    w, f, turns = await match("damage", "damage", "tune1", "eval1", n, conc)
    dt = time.time() - t0; dc = cpu() - c0
    print(f"S3 conc={conc}: {f} battles, {dt:.1f}s wall, {f/dt:.1f} battles/s, {turns/dt:.0f} turns/s (A-side turns), mean turns {turns/f:.1f}, server CPU {dc:.1f}s = {100*dc/dt:.0f}% of one core", flush=True)
async def main():
    await s2(20)
    await s3(400, 1)
    await s3(400, 8)
asyncio.run(main())
