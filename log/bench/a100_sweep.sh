#!/bin/bash -l
# A100 baseline for the MPM particle-count sweep: `default` (the prior torch MPM) vs `warp`
# (the fused refactor). Run UNBUFFERED through the env's own python -- `conda run` buffers stdout
# and a job that prints nothing at the end is indistinguishable from a job that failed.
PY=/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u -c "import torch,warp;print('torch',torch.__version__,'warp',warp.__version__)"

for TAG in 500k 1m 5m 10m 20m 100m; do
  SPEC=config/material/material_3d_water_bench_${TAG}.yaml
  for IMPL in default warp; do
    for CAP in "" "--capture"; do
      echo "### ${TAG} impl=${IMPL} ${CAP:---nocapture}"
      timeout 3600 $PY -u tools/mpm_bench.py --spec $SPEC --sizes 0 --frames 12 --warmup 4 \
        --device cuda:0 --impl $IMPL $CAP --json log/bench/a100_${TAG}_${IMPL}${CAP:+_cap}.json \
        2>&1 | grep -vE "^\[|Warp 1\.|CUDA Toolkit|Devices:|^ +\"c|peer access|Supported|Kernel cache|/.*\.cache/warp"
    done
  done
done
echo "### DONE"
