"""THROWAWAY Showdown spike S4: tune/eval team pools stay disjoint per player, and server battles are not reproducible from the client. Needs a local server and ~/xp-showdown-spike/teams (see the plan)."""
from __future__ import annotations
import asyncio, sys, random, collections
sys.path.insert(0, "/Users/knguyen/VSC/xp-agents/scripts/spikes")
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer
from poke_env.teambuilder import Teambuilder
from showdown_opponents import DamageHeuristicPlayer
FMT = "gen9championsvgc2026regmc"; SUF = "%d" % random.randrange(10**5)
D = "/Users/knguyen/xp-showdown-spike/teams/"
RAW = {n: open(D + n + ".txt").read() for n in ["tune1", "tune2", "tune3", "eval1", "eval2", "eval3"]}
LISTS = {"tune": ["tune1", "tune2", "tune3"], "eval": ["eval1", "eval2", "eval3"]}
assert not set(LISTS["tune"]) & set(LISTS["eval"]) and len({RAW[n] for n in RAW}) == 6, "pools overlap"
class PoolTeam(Teambuilder):
    def __init__(self, names, rng): self.names, self.rng, self.drawn = names, rng, []
    def yield_team(self):
        n = self.rng.choice(self.names); self.drawn.append(n)
        return self.join_team(self.parse_showdown_team(RAW[n]))
async def pool_run(pa, pb, n, seed):
    ta, tb = PoolTeam(LISTS[pa], random.Random(seed)), PoolTeam(LISTS[pb], random.Random(seed + 1))
    mk = lambda cls, nm, tm: cls(account_configuration=AccountConfiguration(nm + SUF, None), battle_format=FMT,
        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1, log_level=30, team=tm)
    a, b = mk(DamageHeuristicPlayer, f"a{pa}", ta), mk(DamageHeuristicPlayer, f"b{pb}", tb)
    await asyncio.wait_for(a.battle_against(b, n_battles=n), 300)
    # observed species sets, per battle, per side, mapped back to a team name
    by_species = {}
    for nm in RAW:
        sp = {s.split("@")[0].strip().split("(")[0].strip().lower().replace("-", "").replace(" ", "") for s in RAW[nm].split("\n\n")}
        by_species[nm] = sp
    seen = {"a": collections.Counter(), "b": collections.Counter()}
    for side, pl in (("a", a), ("b", b)):
      for bt in pl.battles.values():
        for mons in (bt.team,):
            obs = {m.species.replace("mega", "").replace("y", "") if False else m.species for m in mons.values()}
            # a team matches when its 6 base species equal the observed ones (forme-insensitive on the stripped id)
            hit = [nm for nm, sp in by_species.items() if all(any(o.startswith(x[:6]) for o in obs) for x in sp)]
            seen[side][tuple(hit)] += 1
    return ta.drawn, tb.drawn, seen, a.n_finished_battles
async def determinism(n):
    outs = collections.Counter()
    a = MaxBasePowerPlayer(account_configuration=AccountConfiguration("dA" + SUF, None), battle_format=FMT,
        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1, log_level=30, team=RAW["tune1"])
    b = MaxBasePowerPlayer(account_configuration=AccountConfiguration("dB" + SUF, None), battle_format=FMT,
        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1, log_level=30, team=RAW["eval1"])
    await asyncio.wait_for(a.battle_against(b, n_battles=n), 300)
    for bt in a.battles.values(): outs[(bt.won, bt.turn)] += 1
    return outs
async def main():
    for pa, pb in [("tune", "eval"), ("eval", "tune")]:
        da, db, seen, fin = await pool_run(pa, pb, 100, 7)
        ok_a = set(da) <= set(LISTS[pa]) and set(db) <= set(LISTS[pb])
        print(f"A draws from {pa}, B from {pb}: {fin} battles, A drew {dict(collections.Counter(da))}, B drew {dict(collections.Counter(db))}, drew only assigned list: {ok_a}")
        print("  observed in battle objects (side a):", dict(seen["a"]), "| (side b):", dict(seen["b"]))
    outs = await determinism(40)
    print(f"server repeat test: 40 identical MaxBasePower-vs-MaxBasePower battles on fixed teams -> {len(outs)} distinct (won, turns) outcomes: {dict(outs)}")
asyncio.run(main())
