#!/bin/bash -l
# LARGER agent: tools/morph_scale.py, any --exp. Pass every flag through, e.g.
#   bsub ... jobs/morph_scale.sh --exp sweep --pts 12500,50000,200000,500000 --device cuda:0
cd /groups/saalfeld/home/allierc/Graph/Plexus
export PYTHONPATH=/groups/saalfeld/home/allierc/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 MPLBACKEND=Agg
# An L4 is 22 GiB USABLE, not 24 -- the coordinator hit torch.OutOfMemoryError at only 50,000
# points with 347 MiB reserved-but-unallocated (fragmentation), and the error message itself
# recommends this. Matches the same fix applied to jobs/morph.sh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n connectome-gnn python tools/morph_scale.py "$@"
