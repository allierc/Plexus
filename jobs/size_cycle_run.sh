#!/bin/bash -l
# One size/cycle spec on the cluster, archived under log/size_cycle/<rung>/graphs_data/tissue/<spec>
# so every rung of notes/size_cycle/SIZE_CYCLE_PLAN.md keeps its own archives.
#     bash jobs/size_cycle_run.sh <spec> <rung>
spec="$1"; rung="$2"
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/Graph/Plexus/log/size_cycle/$rung
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PLEXUS_STRICT_DETERMINISM=1
mkdir -p "$GNN_OUTPUT_ROOT"
conda run -n connectome-gnn python Plexus_Main.py -o generate tissue/$spec --device cuda:0 --force
