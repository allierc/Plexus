#!/bin/bash -l
cd ${CLUSTER_HOME}/Graph/Plexus
export PYTHONPATH=${CLUSTER_HOME}/Graph/Plexus/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PLEXUS_STRICT_DETERMINISM=1
conda run -n connectome-gnn python Plexus_Main.py -o generate tissue/apopgeo_half --device cuda:0 --force --no-describe
