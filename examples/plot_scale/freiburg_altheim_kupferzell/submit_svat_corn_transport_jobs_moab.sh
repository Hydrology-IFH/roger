#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_corn_CCCma-CanESM2_CCLM4-8-17_*_cpu_moab.sh"
for f in $FILES
do
  qsub -q long $f
done

FILES="$PWD/svat_transport_*_corn_MPI-M-MPI-ESM-LR_RCA4_*_cpu_moab.sh"
for f in $FILES
do
  qsub -q long $f
done

FILES="$PWD/svat_transport_*_corn_observed_*_cpu_moab.sh"
for f in $FILES
do
  qsub -q long $f
done