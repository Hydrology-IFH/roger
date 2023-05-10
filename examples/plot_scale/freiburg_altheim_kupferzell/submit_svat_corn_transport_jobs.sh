#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_corn_CCCma-CanESM2_CCLM4-8-17_*.sh"
for f in $FILES
do
  qsub -q short $f
done

FILES="$PWD/svat_transport_*_corn_MPI-M-MPI-ESM-LR_RCA4_*.sh"
for f in $FILES
do
  qsub -q short $f
done

FILES="$PWD/svat_transport_*_corn_observed_*.sh"
for f in $FILES
do
  qsub -q short $f
done