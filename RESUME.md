# XP — Resume

## Nothing in progress (2026-10-28 20:30)

The Showdown full run FINISHED at 20:19 (started 16:37, resumed from cached tuning jobs; HEAD `f27411c`, `ALLOW_CODE_CHANGE=1`, 2 workers). All 20 eval seed jobs are done (10 Kairi, 10 Kaiser, seeds 100 to 109). The report was run: tables and `degradation.png` are in `figs/showdown` (gitignored, like `results/`). No server or workers are left running.

- **Result: a tie on the familiar opponent.** Training AUC: Kairi 0.406 (0.397 to 0.413), Kaiser 0.403 (0.394 to 0.410). `win_eval`: 0.455 vs 0.451. Steps to 30% win rate: Kairi 29,696 (24,576 to 34,816) vs Kaiser 34,816 (31,744 to 37,888), intervals touch. All 20 seeds reached 30%. No crossover (one level only, no shift knob).
- **Tuning:** Kairi `ent_coef` 0.01 (score 0.610), Kaiser `rnd_coef` 0.03 (score 0.612), both interior (`boundary=none`), so nothing is provisional. Score is `1 - AUC`, lower is better.
- **Figure caveat:** `degradation.png` has two dots on a zoomed y-axis (0.395 to 0.413) and no visible error bars, which exaggerates a 0.003 gap. Widen the axis or lead with the tables when writing it up.
- **Swap during the run:** the patched server stayed flat at about 500 MB, but macOS swap still rose to about 6 GB when the Kaiser eval jobs began (RND), then fell back to about 2 GB at the end; free disk stayed above 70 GiB. Cause not isolated; it is probably the training workers, not the server. Keep the 10 s `TIMEOUT_EMPTY_DEALLOCATE` patch (`~/xp-showdown-spike/pokemon-showdown/server/rooms.ts`, see `docs/spikes/showdown_memory.md`).
- **Done:** the Showdown result is in `README.md`, `docs/FINDINGS.md`, `docs/WRITEUP.md`; tables and figure copied to `docs/spikes/showdown_results/`. **Not yet done:** `docs/findings.html` (built from `scripts/findings_template.html`) has no Showdown section; push (needs the user's yes); the battle-replay feature.
- **Next, when the user says go:** battle replays, plan in `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md` (deferred by the user until after the full run and its testing; needs a yes before building; it retrains agents, since policies are not saved).
- **If a run ever needs resuming:** same command as before (`ALLOW_CODE_CHANGE=1 caffeinate -i scripts/run_showdown.sh configs/showdown.yaml results/showdown 2`), check swap first, kill orphans with `pgrep -fl "xp.run|multiprocessing.spawn|pokemon-showdown|room-battle"`. `pkill -f` patterns can match the calling shell.

## Settled facts

- **User decisions (2026-10-26):** familiar opponent = DamageHeuristic, cap S = 250,000 decisions per run (spec 5.6 amendment; `docs/spikes/showdown_gate.md`), threshold 0.30. Plan C: `docs/superpowers/plans/2026-10-26-showdown-phase-wiring.md`. The user approves each step before it is done (see the memory note) and was asked right before this launch.
- **Showdown work is committed but NOT pushed** (the user pushes on request); everything else is pushed. Pushes need the user's yes each time.
- **Crafter** closed as a documented spike (2026-10-26): plain PPO reached 4.52 / 4.37 achievements per episode, below the G1 gate of 5. Reopening needs a spec amendment raising S first. See `docs/spikes/crafter_spike.md`.
- **MiniGrid DoorKey** closed (2026-10-24); **MultiRoom confirmation** complete (2026-10-25). Do not reopen either without the user asking.
- Smoke run on the real server passed: `docs/spikes/showdown_smoke.md`. Run `python -m xp.showdown.patch` after any poke-env install.
- Open: the occasional server message `[Invalid choice] There's nothing to choose` / `Can't do anything: The game is over` (cause unknown, never stopped a run).
- **Scratch env (outside the repo, not in git):** `~/xp-showdown-spike/` (Python 3.9 venv with patched `poke-env` 0.10.0, local Node 22.23.4, Showdown checkout, teams in `teams/`). Keep it until the user says to delete it. Re-apply `scripts/spikes/showdown_patch_poke_env.py` after any poke-env reinstall.
- **Start the server by hand (the launcher does it for you):** `export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH; cd ~/xp-showdown-spike/pokemon-showdown && node pokemon-showdown start --no-security` (about 25 s; stop with `pkill -f "pokemon-showdown start"`).
- **Gotchas:** rejected teams make `poke-env` hang silently (validate with `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt`); use short alphanumeric account names.
- Stop hook: `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

Deferred by the user (2026-10-27, "later", after the full run): recording battles to watch in the Showdown replay viewer. Plan (untracked): `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md`. Trained policies are not saved today, so it needs retraining.

Optional longer follow-ups (hours of compute, none approved): other MultiRoom room counts, a bigger budget, more DoorKey seeds. See `docs/FUTURE_WORK.md`.
