#!/bin/bash
# overnight -- run a campaign from round 1 and find out how far it gets.
#
#   an ATTEMPT is a whole campaign, starting clean at round 1.
#   a ROUND is one round inside it. Rounds are NOT cleaned between -- that was the bug that made
#   two successful attempts look like two rounds when both were round 1, the second having
#   deleted the first's runs. campaign/state.json holds the round counter; wiping it resets the
#   campaign, so it is wiped ONCE per attempt and never between rounds.
#
#   Survive SURVIVE_N rounds and the attempt has proved itself, so it keeps rolling to MAX_ROUNDS.
#   Crash before that and the attempt ends; the next one starts clean, up to MAX_ATTEMPTS.
#
# It does not patch code. A crash leaves its traceback in the round log and its last breadcrumb
# in campaign/trace.log, and the campaign state at the moment of death is preserved under
# campaign/_failed_<attempt>_<round>/ so the fix can be made from evidence rather than from memory.
set -u
cd /workspace/Plexus/discovery || exit 1
export PYTHONPATH=/workspace/Plexus/src
PY=/workspace/.conda_envs/neural-graph-linux/bin/python

MAX_ATTEMPTS=10
SURVIVE_N=3          # rounds that must pass before the attempt is trusted to roll
MAX_ROUNDS=40        # the overnight ceiling once it has been trusted

clean_campaign() {
    rm -f campaign/{analysis,memory,knowledge,lever_map,causal_descriptions}.md \
          campaign/{proposal,state,frontier}.json \
          campaign/{hypotheses,round_records,archivist,llm_timing,peer_review,diagnoses,supervisor,lever_map}.jsonl \
          campaign/trace.log 2>/dev/null
    rm -rf /workspace/Plexus/log/okuda/r0??n_* /workspace/Plexus/log/okuda/r0??c_* \
           /workspace/Plexus/config/okuda/r0*.yaml /workspace/Plexus/config/okuda/r0*.composition.json 2>/dev/null
}

echo "[overnight] start $(date '+%F %H:%M')  --  up to $MAX_ATTEMPTS attempts, "\
"$SURVIVE_N rounds to earn a roll, $MAX_ROUNDS max"

for a in $(seq 1 $MAX_ATTEMPTS); do
    echo ""
    echo "================================================================================"
    echo "[overnight] ATTEMPT $a/$MAX_ATTEMPTS -- clean campaign, starting at round 1  $(date +%H:%M)"
    echo "================================================================================"
    clean_campaign
    ROUND=0
    while [ "$ROUND" -lt "$MAX_ROUNDS" ]; do
        ROUND=$((ROUND+1))
        LOG=/tmp/on_a${a}_r${ROUND}.log
        echo "[overnight] attempt $a round $ROUND  $(date +%H:%M)  -> $LOG"
        timeout 5400 $PY round.py --mode recon --batch 6 > "$LOG" 2>&1
        RC=$?
        LAST=$(tail -1 campaign/trace.log 2>/dev/null || echo "no trace")
        NREC=$(wc -l < campaign/round_records.jsonl 2>/dev/null || echo 0)
        if [ "$RC" = "0" ]; then
            echo "[overnight]   round $ROUND OK. records=$NREC  last: $LAST"
            if [ "$ROUND" = "$SURVIVE_N" ]; then
                echo "[overnight]   *** $SURVIVE_N rounds survived -- trusted, rolling to $MAX_ROUNDS ***"
            fi
            continue
        fi
        # A round with no admissible evidence (5) is a FINDING, not a crash, and the campaign is
        # entitled to carry on -- round.py stops the run itself after two in a row.
        if [ "$RC" = "5" ]; then
            echo "[overnight]   round $ROUND produced NO EVIDENCE (exit 5) -- continuing; the"\
                 "round itself stops the campaign after two in a row"
            continue
        fi
        echo "[overnight]   round $ROUND FAILED (exit $RC). last trace: $LAST"
        grep -E '^[A-Za-z_.]+(Error|Exception)' "$LOG" | tail -3
        # preserve the evidence BEFORE the next attempt cleans it away
        KEEP=campaign/_failed_a${a}_r${ROUND}
        mkdir -p "$KEEP" 2>/dev/null
        cp campaign/trace.log campaign/round_records.jsonl campaign/analysis.md \
           campaign/memory.md campaign/state.json "$KEEP" 2>/dev/null
        cp "$LOG" "$KEEP/round.log" 2>/dev/null
        echo "[overnight]   state at death preserved in $KEEP"
        break
    done
    if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
        echo "[overnight] ATTEMPT $a reached the $MAX_ROUNDS-round ceiling. Stopping."
        break
    fi
done
echo ""
echo "[overnight] finished $(date '+%F %H:%M')"
