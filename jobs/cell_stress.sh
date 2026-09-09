#!/bin/bash -l
# One active-stress arm, RENDERED: $1 is the amplitude tag, f10 / f25 / f50 -- the active stress
# as a fraction of the cytoskeleton's own (lambda + 2 mu), which is the strain it is asking for.
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
conda run -n connectome-gnn python Plexus_Main.py -o generate "cell/adh_stress_$1" \
    --device cuda:0 --force --render-max-frames 400
