#!/bin/bash -l
# Render one or more morph.npz directories as glass. All args pass straight through to
# tools/morph_render.py, e.g.:
#   bsub -n 8 -gpu num=1 -q gpu_l4 -W 30 jobs/morph_render.sh --dirs graphs_data/si_material/morph_armadillo --compare
#   bsub -n 8 -gpu num=1 -q gpu_l4 -W 60 jobs/morph_render.sh --glob 'graphs_data/si_material/morph_*' --compare
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
conda run -n connectome-gnn python tools/morph_render.py "$@"
