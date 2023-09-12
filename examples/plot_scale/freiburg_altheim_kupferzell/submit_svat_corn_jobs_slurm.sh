#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat

FILES="$PWD/svat_*_corn_CCCma-CanESM2_CCLM4-8-17_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done

FILES="$PWD/svat_*_corn_MPI-M-MPI-ESM-LR_RCA4_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done