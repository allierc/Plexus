#!/bin/bash
# The rung on this machine's two GPUs when the cluster is out of reach: one spec per GPU at a time,
# archived exactly where the cluster script puts them (log/size_cycle/<rung>/graphs_data/tissue).
#     bash jobs/size_cycle_local.sh <rung>
rung="$1"; [ -z "$rung" ] && { echo "usage: $0 <rung>"; exit 1; }
cd /workspace/Plexus
export PYTHONPATH=/workspace/Plexus/src GNN_OUTPUT_ROOT=/workspace/Plexus/log/size_cycle/$rung
export MPLBACKEND=Agg PLEXUS_STRICT_DETERMINISM=1 OMP_NUM_THREADS=8
P=/workspace/.conda_envs/neural-graph-linux/bin/python
mkdir -p "$GNN_OUTPUT_ROOT" jobs/logs/size_cycle
run() { $P Plexus_Main.py -o generate tissue/$1 --device $2 --force > jobs/logs/size_cycle/${rung}_$1.out 2>&1; echo "done $1 $(date +%H:%M)"; }
A="size_sizer size_adder size_doubler size_timer size_grow_sizer mech_target_percell"
B="size_two_channel cycle_sizer cycle_timer cycle_hazard cycle_dilution"
(for s in $A; do run $s cuda:0; done) &
(for s in $B; do run $s cuda:1; done) &
wait
echo "all done $(date +%H:%M)"
