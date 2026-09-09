#!/bin/bash -l
# Does the control need to ROTATE material, or is a pure stretch enough?
#
# $1 = target name (an .obj in papers/morph_models), $2 = 6 or 9 components per control node,
# $3 = output directory. Everything else is held fixed between the two arms so the only difference
# is whether the rate tensor A may have an antisymmetric part.
#
# 6 components store a symmetric A, so expm(-A dt) is symmetric positive definite: a pure stretch
# along principal axes. A material point may lengthen and thin but its rest configuration can never
# turn. 9 components store the full 3x3, whose antisymmetric part is a rotation rate. The elastic
# energy reads F_e = F F_g^-1, so a rotation living inside the growth part F_g survives into
# F_e^T F_e and does real work -- it is not a gauge freedom that cancels.
#
# Prediction if the hypothesis is right: 9 wins on targets whose mass must sweep sideways out of the
# body (a teapot's spout and handle, an armadillo's limbs) and ties on targets that are mostly a
# stretched blob (spot).
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
# A 50,000-point 20-frame differentiable tape does not fit an L4's 22 GiB -- measured, and
# expandable_segments does not save it, because the memory is live tape tensors and not
# fragmentation. These jobs go to gpu_a100 (80 GiB), where the same run peaks at 27.8 GiB compiled.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n connectome-gnn python tools/morph_gallery.py \
    --targets "$1" --control grid --ctrl 12 --control-dof "${2:-6}" \
    --stages "5000:24:80,12500:32:80,50000:40:80" \
    --frames 20 --render-frames 100 --lr 3.0 --device cuda:0 \
    --out "${3:-/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/dof}"
