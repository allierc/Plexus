#!/bin/bash -l
# One case of the adhesion sweep. $1 is the case index into tools/cell_adhesion_sweep.py CASES.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
conda run -n connectome-gnn python tools/cell_adhesion_sweep.py \
    --device cuda:0 --only "$1" --frames 900 --divide 2 --n-grid 224 --substep 4e-5 \
    --out /groups/saalfeld/home/allierc/Graph/Plexus/log/cell/adhesion
