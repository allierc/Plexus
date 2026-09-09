#!/bin/bash -l
# One morph_physics.py experiment. $1 = tag (output subfolder + results_<tag>.json under
# graphs_data/si_material/more_precise/), all remaining args pass straight through as flags, e.g.:
#   bsub ... jobs/morph_physics.sh tear --exp tear --stages 5000:24:80,12500:32:80,50000:40:80
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
# gpu_l4 carries 22 GiB (the baseline in si_material/README.md was measured on a 48 GiB RTX
# A6000); a 3-stage coarse-to-fine run recompiles+recaptures a CUDA graph per stage, and
# `expandable_segments` cuts fragmentation from the resulting allocate/free churn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG="$1"; shift
conda run -n connectome-gnn python tools/morph_physics.py --tag "$TAG" --device cuda:0 "$@"
