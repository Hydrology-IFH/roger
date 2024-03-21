#!/bin/bash

cd ~/roger/examples/plot_scale/boadkh/svat_crop_nitrate

# FILES="$PWD/svat_crop_nitrate_freiburg_*_slurm.sh"
# for f in $FILES
# do
#   sbatch --partition=single $f
# done

FILES="$PWD/svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_*_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done