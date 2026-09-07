#!/bin/bash -l
# One cell spec, RENDERED: $1 is the spec name under config/cell, e.g. adh_grow_all.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
conda run -n connectome-gnn python Plexus_Main.py -o generate "cell/$1" \
    --device cuda:0 --force --render-max-frames 400
