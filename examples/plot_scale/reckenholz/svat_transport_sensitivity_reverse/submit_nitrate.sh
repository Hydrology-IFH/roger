#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_transport_sensitivity_reverse
FILES="$PWD/nitrate_*.sh"
for f in $FILES
do
  sbatch ./$f
done
