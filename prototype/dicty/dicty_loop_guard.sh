#!/usr/bin/env bash
# Liveness guard for the dicty agentic loop overnight: if dicty_loop.py dies (and the run isn't
# finished), relaunch it RESUMING from saved state. Bounded to HOURS so it can't run away.
# Quality/safety is handled separately by the hourly examiner cron. Stop early:
#   pkill -f dicty_loop_guard; pkill -f dicty_loop.py; pkill -f "claude -p"
set -u
DIR=/workspace/Plexus/prototype/dicty
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
EP=/workspace/Plexus/src:/workspace/Plexus/prototype
HOURS=13
N_BATCHES=40
DEADLINE=$(( $(date +%s) + HOURS*3600 ))
cd "$DIR"
echo "[guard] start $(date), ${HOURS}h window" >> dicty_loop_guard.log
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  BATCH=$(${PY} -c "import json;print(json.load(open('dicty_loop_state.json'))['batch'])" 2>/dev/null || echo 1)
  if [ "$BATCH" -gt "$N_BATCHES" ]; then
    echo "[guard] $(date): all $N_BATCHES batches done (batch=$BATCH) -- exiting guard" >> dicty_loop_guard.log
    break
  fi
  if ! pgrep -f "dicty_loop.py" >/dev/null; then
    echo "[guard] $(date): loop down at batch $BATCH -> relaunch --resume" >> dicty_loop_guard.log
    PYTHONPATH="$EP" EVAL_DEVICE=cuda:0 CLAUDE_TIMEOUT_MIN=30 \
      nohup "$PY" dicty_loop.py "$N_BATCHES" --resume >> dicty_loop_run.log 2>&1 &
    sleep 30
  fi
  sleep 300
done
echo "[guard] stopped $(date)" >> dicty_loop_guard.log
