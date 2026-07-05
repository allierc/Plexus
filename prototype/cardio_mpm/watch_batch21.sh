#!/bin/bash
# Wait for batch-21 jobs to genuinely finish (positive evidence), then relaunch the
# hardened cardio loop at state 22 so design-22 analyzes COMPLETE batch-21 results.
cd /workspace/Plexus/prototype/cardio_mpm
JOBS="151960618 151960619 151960620 151960621"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 allierc@login1"
while true; do
  out=$(timeout 90 $SSH "bash -l -c 'source /etc/profile.d/profile.lsf.sh; bjobs -o \"jobid stat\" $JOBS 2>/dev/null'" 2>/dev/null)
  running=$(echo "$out" | grep -cE "RUN|PEND")
  terminal=$(echo "$out" | grep -cE "DONE|EXIT")
  echo "$(date '+%m-%d %H:%M') running=$running terminal=$terminal"
  # break ONLY on positive evidence: valid non-empty response, 0 running, >=1 terminal
  if [ -n "$out" ] && [ "$running" = "0" ] && [ "$terminal" -ge 1 ]; then break; fi
  sleep 300
done
echo "$(date '+%m-%d %H:%M') batch 21 COMPLETE -> relaunching hardened loop at state $(cat cardio_mpm_loop_state.json)"
nohup /workspace/.conda_envs/neural-graph-linux/bin/python cardio_mpm_loop.py 40 > loop_logs/resume3.out 2>&1 &
echo "relaunched loop PID $!"
