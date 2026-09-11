#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PLEXUS_STRICT_DETERMINISM=1
conda run -n connectome-gnn python Plexus_Main.py -o generate tissue/ab_11_plane --device cuda:0 --force --no-describe
