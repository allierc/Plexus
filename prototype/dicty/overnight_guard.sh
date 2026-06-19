#!/usr/bin/env bash
# Bounded crash-restart guard for the dicty optimizer overnight.
# Every 5 min until DEADLINE: if no optimizer is running, relaunch it RESUMING from the
# saved state (no --reset, so the accumulated UCB history is kept). Exits at DEADLINE so
# it can never run away. Stop everything early with: pkill -f opt_dicty_continue; pkill -f overnight_guard
set -u
DIR=/workspace/Plexus/prototype/dicty
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
EP=/workspace/Plexus/src:/workspace/Plexus/prototype
HOURS=13                                  # guard window (covers the 12h run + buffer)
DEADLINE=$(( $(date +%s) + HOURS*3600 ))
cd "$DIR"
echo "[guard] start $(date), deadline in ${HOURS}h" >> guard.log
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f "opt_dicty_continue.py" >/dev/null; then
    echo "[guard] $(date): optimizer not running -> relaunch (resume)" >> guard.log
    PYTHONPATH="$EP" DEVICE=cuda:1 nohup "$PY" opt_dicty_continue.py 43200 >> opt_continue.log 2>&1 &
    sleep 30
  fi
  sleep 300
done
echo "[guard] deadline reached $(date) -- no more restarts" >> guard.log
