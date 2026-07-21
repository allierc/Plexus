#!/bin/bash -l
cd /groups/saalfeld/home/allierc/Graph/Plexus/prototype/Tyssue
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
conda run -n connectome-gnn python run_tyssue_flow.py --only flow_solid_v12_l35
