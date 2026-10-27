#!/usr/bin/env bash
# Run, or resume, the Showdown phase: starts the local server if it is not up, keeps the Mac awake, logs to <out>/run.log.
# Usage: scripts/run_showdown.sh [config] [out_dir] [workers]      (defaults: configs/showdown.yaml results/showdown 2)
# Resumable: every tuning and eval job is cached, so rerunning the same command continues where it stopped.
# NOTE: a resume refuses to run if git HEAD moved since the run started (even a docs commit); do not commit during a run,
# or resume with ALLOW_CODE_CHANGE=1 if the change is deliberate.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/showdown.yaml}
OUT=${2:-results/showdown}
WORKERS=${3:-2}
SPIKE=${XP_SPIKE_DIR:-$HOME/xp-showdown-spike}
export PATH="$SPIKE/node-v22.23.4-darwin-arm64/bin:$PATH"
export XP_NODE="$(command -v node)" XP_SHOWDOWN_DIR="$SPIKE/pokemon-showdown"
source .venv/bin/activate
python -m xp.showdown.patch > /dev/null

STARTED=0
if ! lsof -iTCP:8000 -sTCP:LISTEN > /dev/null 2>&1; then
  (cd "$XP_SHOWDOWN_DIR" && nohup node pokemon-showdown start --no-security > "$SPIKE/server.log" 2>&1 &)
  STARTED=1
  until lsof -iTCP:8000 -sTCP:LISTEN > /dev/null 2>&1; do sleep 2; done
fi
cleanup() { if [ "$STARTED" = 1 ]; then pkill -f "pokemon-showdown start" || true; fi; }
trap cleanup EXIT

mkdir -p "$OUT"
KEEP_AWAKE=""
if command -v caffeinate > /dev/null 2>&1; then KEEP_AWAKE="caffeinate -i"; fi
EXTRA=""
if [ "${ALLOW_CODE_CHANGE:-0}" = 1 ]; then EXTRA="--allow-code-change"; fi
echo "[$(date '+%F %T')] start $CFG -> $OUT (workers $WORKERS)" | tee -a "$OUT/run.log"
$KEEP_AWAKE python -u -m xp.run --phase showdown --config "$CFG" --out "$OUT" --workers "$WORKERS" $EXTRA 2>&1 | tee -a "$OUT/run.log"
echo "[$(date '+%F %T')] finished" | tee -a "$OUT/run.log"
