#!/bin/bash
# Submit the eye characterisation to an L4 cluster.
#
# The repo is on the shared filesystem, so a cluster node sees this prototype at
#   /groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
# (inside the devcontainer the same path is /workspace/Plexus/prototype/eye).
# Nothing here writes outside the archive, and every run is independent, so the
# 112 runs of the protocol shard by run with no communication.
#
#   ./submit_l4.sh derisk                 # the two reduction tests, 3 runs
#   ./submit_l4.sh stage1 F               # the 30 stage-1 holds, one job per muscle
#
# One job per GPU; ask for more slots than muscles and they simply queue.
set -euo pipefail
ROOT=/groups/saalfeld/home/allierc/Graph/Plexus
EYE=$ROOT/prototype/eye
PY=/workspace/.conda_envs/neural-graph-linux/bin/python      # adjust for the cluster image
WHAT=${1:-derisk}
MODEL=${2:-F}

submit () {                       # submit <name> <command...>
  local name=$1; shift
  # bsub keeps RELATIVE paths (the cluster's cwd differs from the devcontainer's)
  bsub -J "eye_${name}" -n 4 -gpu "num=1" -q gpu_l4 \
       -o "$EYE/archive/logs/${name}.out" -e "$EYE/archive/logs/${name}.err" \
       "cd $EYE && $*"
}

mkdir -p "$EYE/archive/logs"
case "$WHAT" in
  derisk)
    for v in baseline substep15 grid96; do
      submit "derisk_$v" "$PY derisk_tests.py --variants $v --device cuda:0"
    done ;;
  stage1)
    for m in LR MR SR IR SO IO; do
      submit "stair_${MODEL}_${m}" \
        "$PY run_staircase.py --model $MODEL --muscles $m --levels 1.0 0.75 0.5 0.25 0.1 --device cuda:0"
    done ;;
  *) echo "unknown target: $WHAT" >&2; exit 2 ;;
esac
