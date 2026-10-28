# Showdown battle replays (plan, for AFTER the full run)

**Status:** proposed, not approved, not built. Do not start until the full run has finished (a commit during the run changes HEAD and breaks the resume check). Needs the user's yes before building.

**Goal:** record a few battles per agent and watch them later in the Showdown replay viewer, to see qualitatively how Kairi and Kaiser play.

## What exists (checked 2026-10-27)

- No code under `xp/showdown/` saves replays or battle logs. The local server has `logchat = false` and its `logs/` folder holds no battle logs.
- **Trained policies are not saved.** `seed_job` (`xp/showdown/phase.py`) trains a `Learner` on the eval pool, evaluates it, writes scores plus a curve JSON, and discards the policy. There is no `torch.save` or checkpoint anywhere in `xp/showdown`, `xp/minigrid` or `xp/protocol/jobs.py`.
- So the recorded battles cannot come from the full run's agents. They need fresh training.

## Design

Keep it out of the research pipeline: a separate script, not a change to `seed_job`, `run_phase` or any hashed config.

1. **`scripts/record_showdown_replays.py`** (new). Arguments: agent (`kairi` or `kaiser`), the tuned hyperparameter (read from the finished `results/showdown` tuning output, never typed by hand), a training seed from the eval pool, number of battles, out dir.
2. Train one agent with `build_learner` + `make_env(cfg, "eval", seed)` for the configured budget (about 30 to 60 min per agent, same as one seed job), then freeze it.
3. Play N battles (suggest 5 per agent, against the familiar opponent) with the frozen policy and keep each battle's protocol log. poke-env holds the message log on each battle object; confirm the exact attribute in the installed 0.10.0 before coding, and write it as `<agent>_seed<S>_battle<k>.log`.
4. Save the frozen policy (`torch.save` of the state dict) next to the logs so later recordings do not need retraining.
5. **Viewer:** a small generator that wraps each log in a standalone replay HTML page. Prefer the viewer served from the local Showdown checkout (`~/xp-showdown-spike/pokemon-showdown`, works offline); fall back to the official replay page script if needed. Output to `replays/` (gitignored, large, not research data). Verify one page opens in a browser (Claude in Chrome can do this) before claiming it works.

## Open questions to resolve first

- Whether the Champions-only species and Mega mechanic render in the stock viewer (this checkout is patched; the stock viewer may lack the sprites). If not, logs still replay, with missing sprites.
- Whether poke-env's battle log includes both players' views (needs the spectator-style log for the viewer).
- Whether to pick the same seeds as the full run (so recorded agents match a scored seed in `results/showdown`) or just any eval-pool seed. Retraining with the same seed is deterministic per config but not byte-identical to the saved run; say so in the write-up.

## Rules

- Recording must not change any result: no edits to the hashed config, `seed_job`, or the results store; nothing written into `results/showdown`.
- Use eval-pool teams and eval seeds only (disjoint from tune seeds; the code raises `ProtocolError` otherwise).
- Tests: add a fast unit test for the log-to-HTML wrapper; the recording itself needs the server (`pytest -m showdown`).
- Server: start it by hand or via `scripts/run_showdown.sh`'s helper; Node 22 path is in `RESUME.md`.

## Steps and rough cost

1. Check poke-env's battle log attribute and the viewer options (30 min, no compute).
2. Write the script and wrapper, with tests (1 to 2 h of work).
3. Smoke test with a tiny budget on the real server (minutes).
4. Real recording: 2 agents x (1 training run + 5 battles), about 1 to 2 h of compute.
5. Open the replays in a browser and check them together.
