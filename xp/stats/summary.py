from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xp.stats.ci import bootstrap_ci, paired_diff_ci


def _values(df: pd.DataFrame, metric: str = "regret") -> Dict[Tuple[str, float], np.ndarray]:
    """{(agent, knob): per-seed values sorted by seed}; checks every cell has the same seeds."""
    out: Dict[Tuple[str, float], np.ndarray] = {}
    seed_sets = []
    for (agent, knob), g in df[df["metric"] == metric].groupby(["agent", "knob"]):
        g = g.sort_values("seed")
        out[(agent, float(knob))] = g["value"].to_numpy(dtype=float)
        seed_sets.append(tuple(g["seed"]))
    if len(set(seed_sets)) > 1:
        raise ValueError("every (agent, knob) must cover the same eval seeds to be paired")
    return out


def curve_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0, metric: str = "regret") -> pd.DataFrame:
    rows = []
    for (agent, knob), v in sorted(_values(df, metric).items()):
        mean, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
        rows.append({"agent": agent, "knob": knob, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def gap_table(df: pd.DataFrame, familiar: float, n_boot: int = 2000, seed: int = 0, metric: str = "regret") -> pd.DataFrame:
    vals = _values(df, metric)
    rows = []
    for (agent, knob), v in sorted(vals.items()):
        ref = vals[(agent, float(familiar))]
        mean, lo, hi = paired_diff_ci(v, ref, n_boot=n_boot, seed=seed)
        rows.append({"agent": agent, "knob": knob, "gap": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _winner(lo: float, hi: float, a: str, b: str, lower_is_better: bool = True) -> str:
    """CI [lo, hi] is for mean(a - b). With lower-is-better a negative difference favours a; higher-is-better flips it."""
    if hi < 0:
        return a if lower_is_better else b
    if lo > 0:
        return b if lower_is_better else a
    return "tie"


def pair_table(
    df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0, metric: str = "regret",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    vals = _values(df, metric)
    knobs = sorted({k for (_, k) in vals})
    rows = []
    for knob in knobs:
        mean, lo, hi = paired_diff_ci(vals[(a, knob)], vals[(b, knob)], n_boot=n_boot, seed=seed)
        rows.append({"knob": knob, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, a, b, lower_is_better)})
    return pd.DataFrame(rows)


def crossings(pair: pd.DataFrame) -> List[Tuple[float, float]]:
    decided = [(float(r.knob), r.winner) for r in pair.sort_values("knob").itertuples() if r.winner != "tie"]
    return [(k0, k1) for (k0, w0), (k1, w1) in zip(decided, decided[1:]) if w0 != w1]


def pair_vs_best_single(
    df: pd.DataFrame, team: str, singles=("kairi", "kaiser"), n_boot: int = 2000, seed: int = 0,
    metric: str = "regret", lower_is_better: bool = True,
) -> pd.DataFrame:
    """Compare `team` with whichever single agent is better (by mean `metric`) at each knob."""
    vals = _values(df, metric)
    knobs = sorted({k for (_, k) in vals})
    rows = []
    for knob in knobs:
        pick = min if lower_is_better else max
        best = pick(singles, key=lambda s: vals[(s, knob)].mean())
        mean, lo, hi = paired_diff_ci(vals[(team, knob)], vals[(best, knob)], n_boot=n_boot, seed=seed)
        rows.append(
            {"knob": knob, "best_single": best, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, team, best, lower_is_better)}
        )
    return pd.DataFrame(rows)
