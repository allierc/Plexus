#!/bin/bash
# watchdog -- stop the campaign on a failure it cannot recover from, and say why.
#
# Written for an unattended overnight run. Claude only runs when invoked, so a check that lives in
# a conversation turn is not a guarantee; this is. It kills on TWO signatures and nothing else,
# because a watchdog that stops a healthy campaign is worse than no watchdog:
#
#   STALLED   no new run directory for 75 minutes. A round is ~21 min, so this is three missed
#             rounds -- the campaign is wedged, not slow.
#   DEAD      two CONSECUTIVE rounds where no run produced metrics. One dead round happened in the
#             last campaign (r002) and it recovered on its own; two in a row is systematic.
#
# It does NOT kill on: refused slots, broken premises, railed metrics, or a round that learns
# nothing. Those are science, and stopping for them would be the strong-judge failure again.
set -u
LOG=/workspace/Plexus/discovery_okuda/watchdog.log
RUNS=/workspace/Plexus/log/okuda
REC=/workspace/Plexus/discovery_okuda/campaign/records.jsonl
PY=/workspace/.conda_envs/neural-graph-linux/bin/python

say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

pid_of() { pgrep -f "round.py --rounds" | head -1; }

say "watchdog up, guarding pid $(pid_of)"
last_new=$(date +%s)
seen=0

while true; do
    sleep 300
    pid=$(pid_of)
    if [ -z "$pid" ]; then say "campaign process is gone -- watchdog exiting"; exit 0; fi

    # --- STALLED: newest run directory older than 75 min
    newest=$(find "$RUNS" -maxdepth 1 -type d -name 'r0*' -newermt '-75 minutes' 2>/dev/null | head -1)
    if [ -n "$newest" ]; then last_new=$(date +%s); fi
    if [ $(( $(date +%s) - last_new )) -gt 4500 ]; then
        say "STALLED: no new run directory in 75 min. Killing $pid."
        kill "$pid"; exit 1
    fi

    # --- DEAD: two consecutive rounds with no metrics anywhere
    dead=$($PY - <<'EOF' 2>/dev/null
import json, collections
try:
    rows=[json.loads(l) for l in open("/workspace/Plexus/discovery_okuda/campaign/records.jsonl") if l.strip()]
except Exception:
    print(0); raise SystemExit
by=collections.defaultdict(list)
for r in rows: by[r.get("round","?")].append(r)
run=0; worst=0
for rd in sorted(by):
    if any(x.get("metrics") for x in by[rd]): run=0
    else: run+=1; worst=max(worst,run)
print(worst)
EOF
)
    if [ "${dead:-0}" -ge 2 ]; then
        say "DEAD: two consecutive rounds produced no metrics. Killing $pid."
        kill "$pid"; exit 2
    fi

    n=$(ls -d $RUNS/r0*/ 2>/dev/null | wc -l)
    if [ "$n" != "$seen" ]; then say "ok -- $n run dirs, campaign alive (pid $pid)"; seen=$n; fi
done
