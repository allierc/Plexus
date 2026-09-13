#!/bin/bash
# Submit one rung's specs to gpu_l4, all in parallel.
#     bash jobs/size_cycle_submit.sh <rung> [spec ...]      (run on login1, from the repo root)
# With no spec list, the eleven size/cycle specs of notes/size_cycle/SIZE_CYCLE_PLAN.md.
rung="$1"; shift; [ -z "$rung" ] && { echo "usage: $0 <rung> [spec ...]"; exit 1; }
specs="$@"
[ -z "$specs" ] && specs="size_sizer size_adder size_doubler size_timer size_grow_sizer size_two_channel cycle_sizer cycle_timer cycle_hazard cycle_dilution mech_target_percell"
mkdir -p jobs/logs/size_cycle
for s in $specs; do
  bsub -n 8 -gpu "num=1" -R "hname!=e11u12" -R "hname!=h08u02" -q gpu_l4 -W 240 -J ${rung}_$s \
       -o jobs/logs/size_cycle/${rung}_$s.out -e jobs/logs/size_cycle/${rung}_$s.err \
       bash -l jobs/size_cycle_run.sh $s $rung
done
