#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_monte_carlo
FILES="$PWD/*_mc.sh"
for f in $FILES
do
  sbatch ./$f
done
