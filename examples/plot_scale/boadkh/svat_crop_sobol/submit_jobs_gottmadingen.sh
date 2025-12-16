#!/bin/bash

cd ~/roger/examples/plot_scale/boadkh/svat_crop_sobol

FILES="$PWD/svat_crop_gottmadingen_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done