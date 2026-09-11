#!/bin/bash
# Submit the eleven size/cycle specs of one rung to gpu_l4, all in parallel.
#     bash jobs/size_cycle_submit.sh <rung>        (run on $CLUSTER_SSH, from the repo root)
rung="$1"; [ -z "$rung" ] && { echo "usage: $0 <rung>"; exit 1; }
mkdir -p jobs/logs/size_cycle
for s in size_sizer size_adder size_doubler size_timer size_grow_sizer size_two_channel \
         cycle_sizer cycle_timer cycle_hazard cycle_dilution mech_target_percell; do
  bsub -n 8 -gpu "num=1" -R "hname!=<node>" -R "hname!=<node>" -q gpu_l4 -W 240 -J ${rung}_$s \
       -o jobs/logs/size_cycle/${rung}_$s.out -e jobs/logs/size_cycle/${rung}_$s.err \
       bash -l jobs/size_cycle_run.sh $s $rung
done
