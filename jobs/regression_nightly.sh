#!/bin/bash
# The regression series overnight, on the cluster or on a local card.
#
#   cluster (from the repo root, relative paths as every job here):
#     bsub -q gpu_l4 -gpu 'num=1' -n 8 -W 4:00 -J plexus_regression \
#          -o jobs/logs/regression_%J.out -e jobs/logs/regression_%J.err bash jobs/regression_nightly.sh
#   local:
#     nohup bash jobs/regression_nightly.sh cuda:1 > log/regression/nightly.log 2>&1 &
#
# Appends one record to log/regression/archive.jsonl; read it with
#     python tools/regression_dashboard.py
set -u
cd "$(dirname "$0")/.."
DEV=${1:-cuda:0}
mkdir -p log/regression
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q "^connectome-gnn "; then
  PY="conda run -n connectome-gnn python"
else
  PY=${PYTHON:-python}
fi
$PY tools/regression_run.py --device "$DEV" --note "nightly $(hostname)"
