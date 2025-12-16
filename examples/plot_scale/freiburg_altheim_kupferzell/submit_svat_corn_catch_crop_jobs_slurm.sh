#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat

FILES="$PWD/svat_*_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
  sleep 5
done

FILES="$PWD/svat_*_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
  sleep 5
done