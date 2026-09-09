#!/bin/bash -l
# CELL_MORPH jobs, two modes. $1 selects the mode.
#
#   train <target> <ctrl> <stages> <render_frames>
#       Optimise the K^3 x 6 rate field on the cheap single-material proxy (tools/morph_gallery.py,
#       shared -- not edited here). Writes graphs_data/cell_morph_control/morph_<target>/morph.npz,
#       a directory OWNED by this job (not the shared graphs_data/si_material other agents write
#       morph_* into), so a concurrent morph run on the same target name cannot collide.
#       --frames is FIXED at 20 -- tools/cell_morph.py's `over = frames_active - 1` assumes it.
#
#   run <spec-name>
#       Generate + render an already-built config/cell/<spec-name>.yaml (see tools/cell_morph.py
#       build). Always captions (no --no-describe, per house rule).
#
# jobs/cell_spec.sh and jobs/morph.sh are the patterns this follows.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
# gpu_l4 cards measured at ~22 GiB usable, not the ~48 GiB of the devcontainer's A6000s that
# jobs/morph.sh's own default stages ("...,50000:40:80") were sized against -- that stage OOM'd
# here on the differentiable 50,000-point rollout (torch.compile + full 20-frame autograd tape,
# no CUDA-graph capture since the run is not eager-safe under grad). expandable_segments trims
# fragmentation but does not raise the ceiling, so the default stage list below is capped at
# 20,000 points / grid 32 for the final stage instead of 50,000 / 40.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="$1"; shift

if [ "$MODE" = "train" ]; then
    TARGET="${1:-teapot}"
    CTRL="${2:-12}"
    STAGES="${3:-3000:20:60,8000:28:60,20000:32:60}"
    RENDER_FRAMES="${4:-100}"
    conda run -n connectome-gnn python tools/morph_gallery.py \
        --targets "$TARGET" --control grid --ctrl "$CTRL" --stages "$STAGES" \
        --frames 20 --render-frames "$RENDER_FRAMES" --lr 3.0 --device cuda:0 \
        --out graphs_data/cell_morph_control
elif [ "$MODE" = "run" ]; then
    SPEC="${1:?spec name under config/cell required, e.g. cell_morph_teapot}"
    conda run -n connectome-gnn python Plexus_Main.py -o generate "cell/$SPEC" \
        --device cuda:0 --force --keep-stills --render-stills 8
else
    echo "usage: cell_morph.sh {train|run} ..." >&2
    exit 1
fi
