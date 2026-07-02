#!/bin/bash
cd /workspace/Plexus/prototype/embryogenesis
export EMBRYO_CLUSTER=1 EMBRYO_FRAMES=6000 EMBRYO_STRIDE=8 EMBRYO_WALL_MIN=30 EMBRYO_POLL_SEC=60 CLAUDE_TIMEOUT_MIN=30
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
while true; do
  b=$(python3 -c "import json;print(json.load(open('embryo_loop_state.json')).get('batch',1))" 2>/dev/null || echo 1)
  if [ "$b" -le 40 ] && ! pgrep -f "python -u embryo_loop.py" >/dev/null 2>&1; then
    echo "[watchdog] $(date) restarting driver at batch $b" >> loop_logs/watchdog.log
    "$PY" -u embryo_loop.py 40 >> loop_logs/campaign_l4.log 2>&1
  fi
  sleep 30
done
