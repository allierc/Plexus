#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/prototype/src:/groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
conda run -n connectome-gnn python run_hold.py --folder archive/eye_G --muscles LR SR MR IR SO IO --level 0.044 0.396 0.934 0.059 0.619 0.487 --hold-s 2.835 --stage 6d --no-movie --device cuda:0
