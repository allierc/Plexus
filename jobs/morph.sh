#!/bin/bash -l
# One morph target, optimised and rendered as glass. $1 = target name (an .obj basename in
# papers/morph_models), $2 = render frames, $3 = output directory.
#
# QUEUE: gpu_a100, not gpu_l4. A 50,000-particle 20-frame differentiable tape does not fit an L4's
# 22 GiB -- measured, and PYTORCH_CUDA_ALLOC_CONF=expandable_segments does not save it, because the
# memory is live tape tensors and not fragmentation. The same run peaks at 27.8 GiB compiled on an
# A100's 80 GiB.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n connectome-gnn python -m plexus.morph \
    --targets "$1" --control grid --ctrl 12 \
    --stages "5000:24:80,12500:32:80,50000:40:120" \
    --frames 20 --render-frames "${2:-400}" --lr 3.0 --px 1100 --device cuda:0 \
    --out "${3:-/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material}"
