#!/bin/bash

cd ~/roger/examples/plot_scale/boadkh/svat_crop_nitrate

FILES="$PWD/svat_crop_nitrate_muellheim_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done