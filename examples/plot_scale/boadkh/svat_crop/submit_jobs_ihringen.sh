#!/bin/bash

cd ~/roger/examples/plot_scale/boadkh/svat_crop

FILES="$PWD/svat_crop_ihringen_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done