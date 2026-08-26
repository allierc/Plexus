#!/bin/bash -l
# One density arm of the free-surface level sweep, on a gpu_l4 node. The five arms are submitted
# as five jobs so they run in parallel rather than the ~3 h they take in sequence locally.
PY=/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=src
# `type/name` if a slash is given, else the historical `material/` default -- so the same script
# drives config/material/ and config/si_material/ without a second copy.
S=$1
case "$S" in */*) SPEC="$S" ;; *) SPEC="material/$S" ;; esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### $SPEC  $(date +%H:%M:%S)"
# no `| tail`: tail buffers to EOF and a long job then shows nothing at all until it finishes
$PY -u Plexus_Main.py -o generate "$SPEC" --device cuda:0 \
    --render-n 1000000 --render-max-frames 600 --no-describe 2>&1 \
  | tr '\r' '\n' | grep -vE "^\[generate\].*[0-9]%\|"
echo "### DONE $(date +%H:%M:%S)"
