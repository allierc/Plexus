#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/prototype/src:/groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
conda run -n connectome-gnn python run_hold.py --folder archive/eye_G --muscles IR --level 0.5 --hold-s 2.835 --stage 1 --device cuda:0
