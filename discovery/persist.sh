#!/bin/bash
# persist -- run recon rounds until TWO complete, or ten attempts, whichever comes first.
#
# Not a retry loop dressed up. A round that crashes leaves its traceback in the attempt log and
# its last breadcrumb in campaign/trace.log, and the NEXT attempt starts from clean campaign
# state -- because a half-written round record is worse than none, and the failures so far have
# all been late-stage crashes after the expensive work succeeded.
#
# It does NOT patch code. Fixing is a judgement and stays with a person or an agent that can read
# the traceback; this only keeps attempting and keeps the evidence tidy so the fix is cheap.
cd /workspace/Plexus/discovery || exit 1
export PYTHONPATH=/workspace/Plexus/src
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
DONE=0
for i in $(seq 1 10); do
    [ "$DONE" -ge 2 ] && { echo "[persist] TWO ROUNDS COMPLETE after $((i-1)) attempt(s)"; break; }
    # CLEAN ONLY AFTER A FAILURE. Wiping at the start of every attempt destroyed attempt 1's
    # record -- a completed round, deleted by the next attempt's tidy-up. A half-written record is
    # worse than none; a FINISHED one is the entire product.
    if [ "$DONE" -gt 0 ]; then
        echo "[persist] keeping the completed round's record; starting the next round on top"
    else
    rm -f campaign/{analysis,memory,knowledge,lever_map,causal_descriptions}.md \
          campaign/{proposal,state,frontier}.json \
          campaign/{hypotheses,round_records,archivist,llm_timing,peer_review,diagnoses,supervisor,lever_map}.jsonl \
          campaign/trace.log 2>/dev/null
    rm -rf /workspace/Plexus/log/okuda/r0??n_* /workspace/Plexus/config/okuda/r0*.yaml 2>/dev/null
    fi
    LOG=/tmp/persist_$i.log
    echo "[persist] attempt $i/10 at $(date +%H:%M) -> $LOG"
    timeout 3600 $PY round.py --mode recon --batch 6 > "$LOG" 2>&1
    RC=$?
    LAST=$(tail -1 campaign/trace.log 2>/dev/null)
    if [ "$RC" = "0" ]; then
        DONE=$((DONE+1))
        echo "[persist] attempt $i COMPLETED (round $DONE of 2). last trace: $LAST"
    else
        echo "[persist] attempt $i exited $RC. last trace: $LAST"
        grep -E '^[A-Za-z_.]*Error|Exception' "$LOG" | tail -2
    fi
done
echo "[persist] finished: $DONE round(s) completed"
