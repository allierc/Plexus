#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/prototype/src:/groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
conda run -n connectome-gnn python run_hold.py --folder archive/eye_G --muscles LR SR MR IR SO IO --level 0.521 0.027 0.563 0.141 0.177 0.441 --hold-s 2.835 --stage 6d --no-movie --device cuda:0
