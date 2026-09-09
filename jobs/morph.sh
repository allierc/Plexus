#!/bin/bash -l
# One morph target, optimised and rendered. $1 = target name (an .obj in papers/morph_models),
# $2 = control nodes per axis, $3 = stages, $4 = points in the final render, $5 = render frames.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
# THE TAPE IS WHAT FILLS THE CARD, not the model: a differentiable rollout at 50,000 points over 20
# frames needs more than an L4 22 GB, measured -- every gallery job died with CUDA OOM while the
# same run fits a 48 GB A6000. Fragmentation is a part of it, so let the allocator grow segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n connectome-gnn python tools/morph_gallery.py \
    --targets "$1" --control grid --ctrl "${2:-12}" \
    --stages "${3:-5000:24:80,12500:32:80,50000:40:80}" \
    --frames 20 --render-frames "${5:-100}" --lr 3.0 --device cuda:0
