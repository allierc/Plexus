#!/bin/bash -l
PY=/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### material_3d_water_bench_200m  $(date +%H:%M:%S)"
$PY -u Plexus_Main.py -o generate material/material_3d_water_bench_200m \
    --device cuda:0 --render-n 4000000 --render-max-frames 500 --no-describe 2>&1 \
  | tr '\r' '\n' | grep -E "grid-CFL|grid-ppc|UNDER|captured as a CUDA|live-movie|done:|Error|ms/frame" | tail -8
echo "### DONE $(date +%H:%M:%S)"
