#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_transport_monte_carlo
FILES="$PWD/bromide_*.sh"
for f in $FILES
do
  sbatch ./$f
done
