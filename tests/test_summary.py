from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xp.stats.summary import crossings, curve_table, gap_table, pair_table, pair_vs_best_single

SEEDS = np.arange(200)


def _rows(agent, knob, values):
    return pd.DataFrame(
        {
            "phase": "bandit", "agent": agent, "knob": knob, "seed": SEEDS,
            "metric": "regret", "value": values, "provisional": False, "config_hash": "x",
        }
    )


def _synthetic():
    rng = np.random.default_rng(0)
    shared = {k: rng.normal(size=len(SEEDS)) * 0.05 for k in (0.0, 0.1, 0.2)}
    parts = []
    # A is better at 0.0 and 0.1, B is better at 0.2 (crossover between 0.1 and 0.2)
    means_a = {0.0: 0.2, 0.1: 0.3, 0.2: 0.9}
    means_b = {0.0: 0.5, 0.1: 0.4, 0.2: 0.5}
    for k in (0.0, 0.1, 0.2):
        parts.append(_rows("A", k, means_a[k] + shared[k]))
        parts.append(_rows("B", k, means_b[k] + shared[k]))
    return pd.concat(parts, ignore_index=True)


def test_curve_table_means_and_ordering():
    t = curve_table(_synthetic(), n_boot=300)
    assert set(t.columns) == {"agent", "knob", "mean", "lo", "hi"}
    assert len(t) == 6
    assert (t["lo"] <= t["mean"]).all() and (t["mean"] <= t["hi"]).all()
    a0 = t[(t.agent == "A") & (t.knob == 0.0)].iloc[0]
    assert abs(a0["mean"] - 0.2) < 0.02


def test_gap_table_is_zero_at_familiar_and_positive_for_growing_regret():
    t = gap_table(_synthetic(), familiar=0.0, n_boot=300)
    zero = t[(t.agent == "A") & (t.knob == 0.0)].iloc[0]
    assert zero["gap"] == 0.0 and zero["lo"] == 0.0 and zero["hi"] == 0.0
    big = t[(t.agent == "A") & (t.knob == 0.2)].iloc[0]
    assert big["gap"] > 0.5 and big["lo"] > 0.5


def test_pair_table_winners_and_crossing():
    t = pair_table(_synthetic(), "A", "B", n_boot=300)
    assert t.sort_values("knob")["winner"].tolist() == ["A", "A", "B"]
    assert crossings(t) == [(0.1, 0.2)]


def test_crossings_skips_ties_and_reports_none_when_one_agent_dominates():
    t = pd.DataFrame({"knob": [0.0, 0.1, 0.2], "winner": ["A", "tie", "A"]})
    assert crossings(t) == []
    t2 = pd.DataFrame({"knob": [0.0, 0.1, 0.2, 0.3], "winner": ["A", "tie", "tie", "B"]})
    assert crossings(t2) == [(0.0, 0.3)]


def test_pair_vs_best_single_picks_better_single_per_knob():
    df = _synthetic()
    team_vals = np.where(True, 0.35, 0.0) + np.zeros(len(SEEDS))
    team = pd.concat([_rows("T", k, team_vals) for k in (0.0, 0.1, 0.2)], ignore_index=True)
    out = pair_vs_best_single(pd.concat([df, team], ignore_index=True), "T", singles=("A", "B"), n_boot=300)
    row0 = out[out.knob == 0.0].iloc[0]  # best single is A (0.2); team (0.35) is worse
    assert row0["winner"] == "A"
    row2 = out[out.knob == 0.2].iloc[0]  # best single is B (0.5); team (0.35) is better
    assert row2["winner"] == "T"


def test_values_rejects_cells_with_different_seeds():
    a = _rows("A", 0.0, np.zeros(len(SEEDS)))
    b = _rows("B", 0.0, np.zeros(len(SEEDS))).iloc[:-1]
    with pytest.raises(ValueError, match="same eval seeds"):
        curve_table(pd.concat([a, b], ignore_index=True), n_boot=50)


def test_higher_is_better_flips_the_winner():
    df = pd.concat([_rows("A", 0.0, 0.9 + np.zeros(len(SEEDS)) + 0.01 * np.sin(SEEDS)),
                    _rows("B", 0.0, 0.1 + np.zeros(len(SEEDS)) + 0.01 * np.sin(SEEDS))], ignore_index=True)
    df["metric"] = "success"
    lower = pair_table(df, "A", "B", n_boot=200, metric="success", lower_is_better=True)
    higher = pair_table(df, "A", "B", n_boot=200, metric="success", lower_is_better=False)
    assert lower.iloc[0]["winner"] == "B" and higher.iloc[0]["winner"] == "A"
