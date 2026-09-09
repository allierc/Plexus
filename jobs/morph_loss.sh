#!/bin/bash -l
# One morph_loss.py variant. $1 = run-name, all remaining args pass straight through as flags,
# e.g.:  bsub ... jobs/morph_loss.sh smooth_l2_3p0 --target cow --smooth-kind l2 --smooth-weight 3.0
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
RUN_NAME="$1"; shift
conda run -n connectome-gnn python tools/morph_loss.py --run-name "$RUN_NAME" --device cuda:0 "$@"
