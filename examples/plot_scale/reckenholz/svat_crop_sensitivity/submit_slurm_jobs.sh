##!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_sensitivity
FILES="$PWD/*_sa.sh"
for f in $FILES
do
  sbatch ./$f
done
