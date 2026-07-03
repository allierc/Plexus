#!/bin/bash
# Watchdog: keep ONE embryo_loop driver alive (resumes from embryo_loop_state.json). The driver
# does design -> submit -> poll -> montage -> rename dirs (current_stage.txt) and force-advances a
# sub-phase after EMBRYO_PHASE_HOURS. This watchdog stops the whole campaign after ~1 week.
cd /workspace/Plexus/prototype/embryogenesis
export EMBRYO_CLUSTER=1 EMBRYO_FRAMES=12000 EMBRYO_STRIDE=16 EMBRYO_WALL_MIN=45 EMBRYO_POLL_SEC=60 \
       CLAUDE_TIMEOUT_MIN=30 EMBRYO_PHASE_HOURS=48
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
CAMPAIGN_H=${EMBRYO_CAMPAIGN_HOURS:-168}                    # ~1 week
[ -f loop_logs/campaign_start.txt ] || date +%s > loop_logs/campaign_start.txt
START=$(cat loop_logs/campaign_start.txt)
while true; do
  now=$(date +%s); el=$(( (now - START) / 3600 ))
  if [ "$el" -ge "$CAMPAIGN_H" ]; then
    echo "[watchdog] $(date) campaign reached ${CAMPAIGN_H}h (~1 week) — stopping" >> loop_logs/watchdog.log
    break
  fi
  b=$($PY -c "import json;print(json.load(open('embryo_loop_state.json')).get('batch',1))" 2>/dev/null || echo 1)
  if [ "$b" -le 300 ] && ! pgrep -f "python -u embryo_loop.py" >/dev/null 2>&1; then
    echo "[watchdog] $(date) restarting driver at batch $b (phase ${el}h into campaign)" >> loop_logs/watchdog.log
    "$PY" -u embryo_loop.py 300 >> loop_logs/campaign_l4.log 2>&1
  fi
  sleep 30
done
