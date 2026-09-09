#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
conda run -n connectome-gnn python Plexus_Main.py -o generate cell/cell_atlas_25_sets \
    --device cuda:0 --force --render-max-frames 240
