#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/prototype/src:/groups/saalfeld/home/allierc/Graph/Plexus/prototype/eye
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
conda run -n connectome-gnn python run_eye_G.py --program pairs --blend "260802_s2_EYE_MUSCLES_MODEL 2.blend" --parts archive/eye_H/blend_parts --k-bone 5000 --out archive/eye_H --label kopt5000 --hold 200 --rest 160 --turns 0 --az 25 --no-movie --device cuda:0
