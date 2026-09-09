#!/bin/bash -l
# Speed-only benchmark of the differentiable MPM shape-morphing rollout (tools/morph_speed.py).
# $1 = --suite value (comma list of SUITE keys, or "all"), default "headline".
# $2.. = forwarded verbatim to morph_speed.py (e.g. --iters 25 --warmup 5 to de-noise a row).
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
# an L4 is 22 GiB usable; the 50,000-point/20-frame differentiable tape OOMs there even before any
# knob in this file is touched (measured -- see results.json). expandable_segments does not raise
# the ceiling, it only stops fragmentation from biting early (the OOM message itself recommends
# it: reserved-but-unallocated was 1.54 GiB on a run that failed 32 MiB short). The 50k-point rows
# that still need room go on gpu_a100 instead (each result row records its own `gpu` field via
# torch.cuda.get_device_name, so a results.json reader can tell which card a number came from).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
suite="${1:-headline}"
[ $# -gt 0 ] && shift
conda run -n connectome-gnn python tools/morph_speed.py \
    --suite "$suite" --device cuda:0 "$@"
