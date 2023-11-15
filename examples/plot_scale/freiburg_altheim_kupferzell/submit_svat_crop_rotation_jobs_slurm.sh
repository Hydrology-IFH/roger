#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat

FILES="$PWD/svat_*_crop_rotation_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
  sleep 5
done