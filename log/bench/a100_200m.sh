#!/bin/bash -l
PY=/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### material_3d_water_bench_200m  $(date +%H:%M:%S)"
# NO `| tail`: tail buffers to EOF, so a 7-hour job showed nothing at all for its first 46 minutes.
# Three movies from ONE simulation -- the trajectory is not stored, so a re-render is impossible.
$PY -u Plexus_Main.py -o generate material/material_3d_water_bench_200m \
    --device cuda:0 --render-n 10000000,50000000,100000000 \
    --render-max-frames 300 --no-describe 2>&1 | tr '\r' '\n' | grep -vE "^\[generate\].*[0-9]%\|" 
echo "### DONE $(date +%H:%M:%S)"
