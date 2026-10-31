# XP — what is left (brainstorm, 2026-10-30)

Every phase has a result and a write-up (bandit, MiniGrid DoorKey, MultiRoom, Showdown original and matchup runs; Crafter closed as a spike). Nothing below is required to call the project finished. Approved by the user on 2026-10-31: the matchup control run (1 worker, running), the finer Kairi `alpha` sweep (done), the polish items (done), a `v1.0` tag after the control run is written up, and the browser check (the user's). Not approved: battle replays and the section 4 extensions. Each item still needs the user's yes before it starts. Longer follow-ups with fuller costs live in `docs/FUTURE_WORK.md`.

Guiding rule from `CLAUDE.md`: a small, finished, well-measured result beats a sprawling one.

## 1. Loose ends (small, worth doing)

| Item | Cost | Notes |
|---|---|---|
| Look at `docs/findings.html` (Chapter 7) and `figs/showdown-matchup/degradation.png` in a browser | 5 minutes of the user's time | The page was built and its script syntax-checked, but never rendered by us (Chrome extension was not connected). The figure has a zoomed y-axis, so read the tables first. |
| Confirm CI is green on the latest push | a minute | The previous pushes passed on Python 3.9 and 3.12. |
| Optional: tag a release (for example `v1.0`) | a minute | Only if the user wants a fixed reference point. |

## 2. The one scientific gap: the matchup control run

**Problem.** The matchup run changed three things at once: the observation (type-matchup inputs), the tuning grids (6 to 8 values each) and the tuned values (Kairi `ent_coef` 0.01 to 0.006, Kaiser `rnd_coef` 0.03 to 1.0). Against the original run, only Kaiser's training AUC rose clearly (+0.022 [+0.012, +0.031], 9 of 10 seeds). We cannot say whether the inputs or the retuning did it.

**Control.** Run the original no-matchup agents on the matchup run's widened 8-value grids (Kairi `ent_coef` 0.001 to 0.1, Kaiser `rnd_coef` 0.01 to 3.0) in a fresh folder. With the existing two runs this gives a clean 2 by 2 (inputs on or off, wide grids), and `scripts/showdown_matchup_compare.py` can be pointed at it.

- **Cost:** about 9 to 10 hours on 1 worker (68 jobs: 48 tuning plus 20 eval, about 7 jobs per hour without the matchup block). No new code beyond a copy of `configs/showdown.yaml` with the wider grids.
- **Run it on 1 worker.** On 2 workers, swap climbed past 5 GB within minutes. Needs the Showdown server restarted (the launcher does it) and `caffeinate`.
- **Risks:** a grid-edge stop (widen and rerun, ask first); a restarted job loses about 10 minutes.
- **Alternative:** skip it and keep the confound as a stated limitation (already written in README, FINDINGS, WRITEUP and `findings.html`).
- **Recommendation:** do it. It is the only open question that changes what we can claim, and it is config only.

## 3. Cheap checks on the bandit phase

- **DONE 2026-10-31: finer Kairi `alpha` sweep, tune seeds only** (best 0.2236 vs 0.2379 on the old grid, Kaiser 0.1724; result in `docs/WRITEUP.md`, script `scripts/bandit_alpha_sweep.py`, table `docs/bandit_alpha_sweep/`). Original note: In the familiar bandit condition Kairi's `epsilon` barely changes the score, so its 49-point grid is effectively one-dimensional while Kaiser's is a genuine two-dimensional valley. A finer `alpha` sweep would show whether that favours Kaiser. Changes the config hash, so it needs a fresh results folder. Quick; no eval seeds are touched.
- **Drift condition:** Kaiser's optimum sits at the natural limit `gamma=1.0` with its nearest neighbour 40% worse. Worth confirming it holds on the eval seeds (the existing results already contain this).

## 4. Optional extensions (not recommended now)

| Item | Cost | Why not now |
|---|---|---|
| Battle-replay recording (40 battles per agent, keep the top 5) | about 1 to 2 hours | The user deferred it. Retrains the agents because policies are not saved. |
| More Showdown follow-ups: tempo features, an opponent that uses Trick Room and Tailwind, probing saved policies for type-chart sensitivity | 12 to 15 hours each | Each is a spec amendment plus a full run. See `docs/FUTURE_WORK.md`. |
| Other MultiRoom room counts, more DoorKey seeds, a bigger budget | hours each | MultiRoom already confirmed Kaiser on the level it was tested on; DoorKey is closed. |
| A deep-RL retune-at-the-shift check (size 8 DoorKey) | config only | Judged low value. |
| Reopen Crafter | days, GPU | Failed its pre-set gate; reopening needs a spec amendment raising the budget S first. |

## 5. Closing out

- Run the fast `pytest` suite once more before any tag (last run passed).
- Keep the scratch environment `~/xp-showdown-spike/` (Python venv with patched `poke-env`, local Node, Showdown checkout) until the user says to delete it; the Showdown server is currently stopped.
- Keep `STATUS.md` and `RESUME.md` current after each step (the Stop hook checks the `STATUS_COMMIT` marker).

## 6. Polish (audit 2026-10-30; the doc-only items are done, the rest wait on the user)

What I checked: repo layout and ignores, README figure links, markdown links, stale phrases, CI, doc sizes. All relative links and README images resolve (0 broken), the fast suite passes, and CI is green. What I found, ordered by value for effort:

| Item | Effort | Why |
|---|---|---|
| ~~**Add a `LICENSE` (and optionally `CITATION.cff`)**~~ **done 2026-10-31 (MIT, Ken Nguyen)** | minutes; needs the user's choice of licence | The repo is public on GitHub with no licence, so by default nobody may reuse it. |
| ~~**Trim `STATUS.md` to the current state**~~ **done 2026-10-30** | 30 minutes | It is 11 KB of dated paragraphs, including superseded "PAUSED" and "RESUMED" notes. Its job is a snapshot; history belongs in git or an archive section. |
| ~~**Trim `RESUME.md`**~~ **done 2026-10-30** | 15 minutes | 16 KB, and it now says "nothing in progress" above a long pile of history. Keep the settled facts (server start, gotchas, scratch environment) and drop the finished runs. |
| ~~**Remove duplication in the write-ups**~~ **done 2026-10-30** | 30 minutes | The matchup paragraph is pasted verbatim into both `README.md` and `docs/FINDINGS.md` (and the FINDINGS heading repeats its bold title). Keep the full text in FINDINGS and give the README a short summary plus a link. |
| ~~**One list of open work**~~ **done 2026-10-30** (this file is the index; STATUS, FINDINGS "Next", FUTURE_WORK and README point here) | 15 minutes | Open items now sit in `STATUS.md`, `docs/FINDINGS.md` "Next", `docs/FUTURE_WORK.md` and this file. Make this file the index and link to it from the others. |
| ~~**Tidy the repo root**~~ **done 2026-10-31** (calibration JSONs now in `configs/`) | 15 minutes | `calibration.json` and `calibration-multiroom.json` sit at the top level; moving them under `configs/` needs the commands in `CLAUDE.md` and the configs updated. Low value; skip unless the root bothers you. |
| ~~**Add a linter to CI** (for example `ruff`)~~ **done 2026-10-31** (rules E9 and F only; 8 unused imports fixed) | 20 minutes | None is installed or run today, so unused imports and similar slips are not caught. Start with a minimal rule set so it does not flood. |
| **Look at the rendered pages** | 5 minutes of the user's time | `docs/findings.html` Chapter 7 and `degradation.png` have never been viewed. Same item as section 1. |
| ~~**Note that CI skips the Showdown server tests**~~ **done 2026-10-30** (README, Reproducing) | 5 minutes | Tests marked `showdown` need a live server and are excluded by design. Say so in the README so it is not mistaken for full coverage. |

Not worth doing now: reformatting the code, renaming packages, or moving the `docs/superpowers/` plans and specs (they are the project's process record).

## Suggested order

1. Browser check of the page and figure (user).
2. Control run overnight on 1 worker, then the paired comparison and a short write-up update.
3. Optional: finer Kairi `alpha` sweep.
4. Polish pass (section 6) while the control run is going; it is doc-only and light on the CPU: licence, trim STATUS and RESUME, deduplicate README and FINDINGS, one list of open work.
5. Tag a release and stop; leave section 4 as listed future work.

## Decisions needed from the user

1. Run the matchup control (section 2), or keep the confound as a limitation?
2. Do the finer Kairi `alpha` sweep (section 3), or leave it as a noted limitation?
3. Tag a release when done?
