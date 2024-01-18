#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_crop_monte_carlo_reference
FILES="$PWD/*_mc_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done
