#!/bin/bash -l
PY=/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for S in material_3d_water_bench_10m material_3d_water_bench_100m; do
  echo "### $S  $(date +%H:%M:%S)"
  $PY -u Plexus_Main.py -o generate "material/$S" --device cuda:0 --render-n 2000000 --no-describe 2>&1 \
    | tr '\r' '\n' | grep -E "grid-CFL|grid-ppc|UNDER|captured as a CUDA|live-movie|done:|Error|ms/frame" | tail -5
done
echo "### DONE $(date +%H:%M:%S)"
